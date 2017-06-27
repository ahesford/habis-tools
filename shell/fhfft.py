#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, os, sys, fht, pyfftw, getopt

from collections import defaultdict, OrderedDict

import multiprocessing

from habis.habiconf import matchfiles, buildpaths
from habis.formats import WaveformSet, loadkeymat
from habis.sigtools import Window
from pycwp import process, mio, cutil

def specwin(nsamp, freqs=None):
	# Ensure default None is encapsulated
	if freqs is None: freqs = (None,)

	# Find the spectral window
	fs, fe, step = slice(*freqs[:2]).indices(nsamp)
	if step != 1:
		raise ValueError('Frequency range must specify consecutive values')

	return Window(fs, end=fe, nonneg=True)


def usage(progname=None, fatal=False):
	if progname is None: progname = sys.argv[0]
	binfile = os.path.basename(progname)
	print >> sys.stderr, 'USAGE: %s [-p p] [-h] [-m tgcmap] [-l n] [-w s:l] [-f s:e:w] [-t] [-n n] [-z] [-b] [-s s] [-o outpath] -g g <measurements>' % binfile
	print >> sys.stderr, '''
  Preprocess HABIS measurement data by Hadamard decoding transmissions and
  Fourier transforming the time-domain data. Measurement data is contained in
  the 'measurements' WaveformSet files.

  Output is stored, by default, in a WaveformSet file whose name has the same
  name as the input with any extension swapped with '.fhfft.wset'. If the input
  file has no extension, an extension is appended.

  REQUIRED ARGUMENTS:
  -g: Use the specified group map to rearrange transmission rows

  OPTIONAL ARGUMENTS:
  -p: Use p processors (default: all available processors)
  -h: Suppress Hadamard decoding
  -m: Use the given 2-column TGC map to convert nominal gain to realized gain
  -l: Apply each TGC value to n samples in the acquisition window (default: 16)
  -f: Retain only DFT bins s:e, with a tail filter of width w (default: all)
  -t: Output time-domain, rather than frequency-domain, waveforms
  -n: Override acquisition window length to n samples in WaveformSet files
  -w: Set start (s, assuming 0 f2c) and length (l) of universal acquisition window
  -z: Subtract a DC bias from the waveforms before processing
  -b: Write output as a simple 3-D matrix rather than a WaveformSet file
  -s: Correct the decoded signs with the given sign map (default: skip)
  -o: Provide a path for placing output files
	'''
	if fatal: sys.exit(fatal)


def _r2c_datatype(idtype):
	'''
	Return the input and output types for a floating-point R2C DFT of an
	input array with data type idtype.
	'''
	if np.issubdtype(idtype, np.complexfloating):
		raise TypeError('Input data type must not be complex')

	# All types except for 32-bit floats are converted to 64-bit floats
	if np.issubdtype(idtype, np.dtype('float32')):
		return np.dtype('complex64')
	else:
		return np.dtype('complex128')


def mpfhfft(nproc, *args, **kwargs):
	'''
	Subdivide, along receive channels, the work of fhfft() among nproc
	processes to Hadamard-decode and Fourier transform the WaveformSet
	stored in infile into a WaveformSet file that will be written to
	outfile.

	The positional and keyward arguments are passed to fhfft(). Any
	'stride', 'start', 'lock', or 'event' kwargs will be overridden by
	internally generated values.
	'''
	if nproc == 1:
		# For a single process, don't spawn
		fhfft(*args, **kwargs)
		return

	# Add the stride to the kwargs
	kwargs['stride'] = nproc

	# Create a multiprocessing lock and event to serialize output access
	kwargs['lock'] = multiprocessing.Lock()
	kwargs['event'] = multiprocessing.Event()

	# Span the desired processes to perform FHFFT
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Give each process a meaningful name
			procname = process.procname(i)
			# Note the starting index for this processor
			kwargs['start'] = i
			pool.addtask(target=fhfft, name=procname, args=args, kwargs=kwargs)
		pool.start()
		pool.wait()


def fhfft(infile, outfile, grpmap, **kwargs):
	'''
	For a real WaveformSet file infile, perform Hadamard decoding and then
	a DFT of the temporal samples. The resulting transformed records will
	be stored in the output file outfile, which will be created or
	truncated. The Hadamard decoding follows the grouping configuration
	stored in the file.

	Outputs will have the transmission indices rearranged according to
	grpmap, a mapping between output transmission i corresponds to input
	transmission

		j = grpmap[i][0] + grpmap[i][1] * wset.txgrps.size

	for wset = WaveformSet.fromfile(infile).

	Any TGC parameters in the input, accessible as wset.context['tgc'],
	will be used to adjust the amplitudes of the waveforms prior to
	applying Hadamard and Fourier transforms.

	The kwargs contain optional values or default overrides:

	* freqs (default: None): When not None, a sequence (start, end) to be
	  passed as slice(start, end) to record only a frequency of interest.

	* nsamp (default: None): When not None, the nsamp property of the input
	  WaveformSet is forced to this value prior to processing.

	* window (default: None): When not None, a sequence (start, length)
	  that specifies the start (relative to a 0 f2c) and length of a global
	  data window that will be extracted from all waveform records. The
	  start must no less than the f2c of each input file because the
	  file-local start of the window (start - wset.f2c) must be nonnegative.

	  This option and nsamp are mutually exclusive.

	* tgcsamps (default: 16 [for integer datatypes] or 0 [else]): The
	  number of temporal samples to which a single TGC parameter applies.
	  Signals will be scaled by an appropriate section of the multiplier

	    mpy = (invtgc[:,np.newaxis] *
		    np.ones((ntgc, tgcsamps), dtype=np.float32)).ravel('C'),

	  where the values invtgc = 10.**(-wset.context['tgc'] / 20.) and
	  ntgc = len(wset.context['tgc']). The multiplier mpy is defined over a
	  window that starts at file sample 0 (global time wset.f2c).

	  Set tgcsamps to 0 (or None) to disable compensation. If the
	  WaveformSet includes TGC parameters and tgcsamps is a positive
	  integer, then len(mpy) must be at least long enough to encompass all
	  data windows encoded in the file.

	* tgcmap (default: None): If provided, should be a two-column, rank-2
	  Numpy array (or compatible sequence) that relates nominal gains in
	  column 0 to actual gains in column 1. The rows of the array will be
	  used as control points in a piecewise linear interpolation (using
	  numpy.interp) that will map TGC parameters specified in the
	  WaveformSet file to actual gains. In other words, the TGC values
	  described above will be replaced with

		tgc = np.interp(tgc, tgcmap[:,0], tgcmap[:,1])

	  whenever tgcmap is provided.

	* dofht (default: True): Set to False to disable Hadamard decoding.

	* tdout (default: False): Set to True to output time-domain waveforms
	  rather than spectral samples. Preserves input acquisition windows.

	* binfile (default: False): Set to True to produce binary matrix
	  output instead of WaveformSet files.

	* signs (default: None): When not None, should be a sequence of length
	  wset.txgrps.size that specifies a 1 for any local Hadamard index
	  (corresponding to lines in the file) that should be negated, and 0
	  anywhere else. Ignored when dofht is False.

	* debias (default: False): If True, all input waveforms will have a DC
	  bias removed prior to processing by subtracting the mean signal value
	  over the sampling window.

	* start (default: 0) and stride (default: 1): For an input WaveformSet
	  wset, process receive channels in wset.rxidx[start::stride].

	* lock (default: None): If not None, it should be a context manager
	  that is invoked to serialize writes to output.

	* event (default: None): If not None, event.set() and event.wait() are
	  called to ensure the output WaveformSet header is written before
	  records are appended. The value event.is_set() should be False prior
	  to execution.
	'''
	# Override acquisition window, if desired
	nsamp = kwargs.pop('nsamp', None)
	window = kwargs.pop('window', None)

	# Enforce exclusivity
	if window and nsamp is not None:
		raise TypeError('Arguments "window" and "nsamp" are mutually exclusive')

	# Grab synchronization mechanisms
	try: lock = kwargs.pop('lock')
	except KeyError: lock = multiprocessing.Lock()
	try: event = kwargs.pop('event')
	except KeyError: event = multiprocessing.Event()

	# Grab FFT and FHT switches and options
	dofht = kwargs.pop('dofht', True)
	tdout = kwargs.pop('tdout', False)
	freqs = kwargs.pop('freqs', None)
	dofft = (freqs is not None) or not tdout

	# Grab output controls
	binfile = kwargs.pop('binfile', False)

	# Grab striding information
	start = kwargs.pop('start', 0)
	stride = kwargs.pop('stride', 1)

	# Grab sign map information
	signs = kwargs.pop('signs', None)

	# Determine whether the waveforms should be debiased
	debias = kwargs.pop('debias', False)

	# Grab the number of samples per TGC value
	tgcsamps = kwargs.pop('tgcsamps', None)

	tgcmap = kwargs.pop('tgcmap', None)

	if len(kwargs):
		raise TypeError("Unrecognized keyword argument '%s'" % kwargs.iterkeys().next())

	# Open the input and create a corresponding output
	wset = WaveformSet.fromfile(infile)

	if not window:
		if nsamp is not None:
			# Force the sample count according to preference
			wset.nsamp = nsamp
		else:
			# Record the sample count according to preference
			nsamp = wset.nsamp

		# Copy the f2c from the input
		f2c = wset.f2c
	else:
		# Validate and unpack the window
		try:
			f2c, nsamp = window
		except (TypeError, ValueError):
			raise ValueError('Argument "window" must have form (start, length)')

	ntx = wset.ntx

	if grpmap is not None:
		# Make sure the WaveformSet has a local configuration
		try:
			gcount, gsize = wset.txgrps
		except TypeError:
			raise ValueError('A valid Tx-group configuration is required')

		# Validate local portion of the group map
		wset.groupmap = grpmap

		# Map global indices to transmission number
		outidx = OrderedDict(sorted((k, v[0] + gsize * v[1]) for k, v in grpmap.iteritems()))
	else:
		# Must specify a group map for FHT decoding
		if dofht: raise ValueError('A valid Tx-group configuration is required')
		# Otherwise, with no configuration, use passthrough transmission ordering
		gsize = 1
		outidx = OrderedDict((i, i) for i in xrange(wset.ntx))


	# Handle TGC compensation if necessary
	try: tgc = np.asarray(wset.context['tgc'], dtype=np.float32)
	except (KeyError, AttributeError): tgc = np.array([], dtype=np.float32)

	if tgcmap is not None:
		# Make sure that the TGC map is sorted and interpolate
		tgx, tgy = zip(*sorted((k, v) for k, v in tgcmap))
		# TGC curves are always float32, regardless of tgcmap types
		tgc = np.interp(tgc, tgx, tgy).astype(np.float32)

	# Pick a suitable default value for tgcsamps
	if tgcsamps is None:
		tgcsamps = 16 if np.issubdtype(wset.dtype, np.integer) else 0

	# Linearize, invert, and expand the TGC curves
	tgc = ((10.**(-tgc[:,np.newaxis] / 20.) *
		np.ones((len(tgc), tgcsamps), dtype=np.float32))).ravel('C')

	if len(tgc):
		# Figure out the data type of compensated waveforms
		itype = np.dtype(wset.dtype.type(0) * tgc.dtype.type(0))
	else:
		itype = wset.dtype

	# Create a WaveformSet object to hold the ungrouped data
	ftype = _r2c_datatype(itype)
	otype = ftype if not tdout else itype
	oset = WaveformSet(len(outidx), 0, nsamp, f2c, otype)
	# Check the output keys for sanity and set the transmit parameters in oset
	oset.txidx = outidx.keys()

	# Prepare a list of input rows to be copied to output
	outrows = [wset.tx2row(i) for i in outidx.itervalues()]

	if dofht:
		if not fht.ispow2(gsize):
			raise ValueError('Hadamard transform length must be a power of 2')

		if signs is not None:
			# Ensure signs has values 0 or 1 in the right type
			signs = np.asarray([1 - 2 * s for s in signs], dtype=itype)
			if signs.ndim != 1 or len(signs) != gsize:
				raise ValueError('Sign array must have shape (wset.txgrps[1],)')

		# Map input transmission groups to local and global indices
		fhts = defaultdict(list)
		for i in wset.txidx:
			gi, li = i // gsize, i % gsize
			fhts[gi].append((li, i))

		# Sort indices in Hadamard index order
		for i, v in fhts.iteritems():
			if len(v) != gsize:
				raise ValueError('Hadamard group %d does not contain full channel set' % i)
			v.sort()
			if any(j != vl[0] for j, vl in enumerate(v)):
				raise ValueError('Hadamard group contains invalid local indices')

	# Create intermediate (FHT) and output (FHFFT) arrays
	# FFT axis is contiguous for FFT performance
	b = pyfftw.n_byte_align_empty((ntx, nsamp),
			pyfftw.simd_alignment, itype, order='C')

	if dofft:
		# Create FFT output and a plan
		cdim = (ntx, nsamp // 2 + 1)
		c = pyfftw.n_byte_align_empty(cdim,
				pyfftw.simd_alignment, ftype, order='C')
		fwdfft = pyfftw.FFTW(b, c, axes=(1,), direction='FFTW_FORWARD')

		if tdout:
			# Create an inverse FFT plan for time-domain output
			invfft = pyfftw.FFTW(c, b, axes=(1,), direction='FFTW_BACKWARD')

		# Find the spectral window of interest
		fswin = specwin(cdim[1], freqs)

		# Try to build bandpass tails
		try: tails = np.hanning(2 * freqs[2])
		except (TypeError, IndexError): tails = np.array([])

	# Create the input file header, if necessary
	with lock:
		if not event.is_set():
			if binfile:
				# Create a sliced binary matrix output
				windim = (fswin.length if dofft else nsamp, oset.ntx, wset.nrx)
				mio.Slicer(outfile, dtype=otype, trunc=True, dim=windim)
			else:
				# Create a WaveformSet output
				oset.store(outfile)
			event.set()

	for rxc in wset.rxidx[start::stride]:
		# Find the input window relative to 0 f2c
		iwin = wset.getheader(rxc).win.shift(wset.f2c)
		owin = (oset.f2c, oset.nsamp)

		try:
			# Find overlap of global input and output windows
			ostart, istart, dlength = cutil.overlap(owin, iwin)
		except TypeError:
			# Default to 0-length windows at start of acquisition
			iwin = Window(0, 0, nonneg=True)
			owin = Window(0, 0, nonneg=True)
		else:
			# Convert input and output windows from global f2c to file f2c
			iwin = Window(istart, dlength, nonneg=True)
			owin = Window(ostart, dlength, nonneg=True)

		# Read the data on input and convert the header window to output
		hdr, data = wset.getrecord(rxc, window=iwin)
		hdr = hdr.copy(win=owin)

		if len(tgc) and iwin.length:
			# Time-gain compensation
			owin = (0, len(tgc))
			try:
				ostart, istart, dlength = cutil.overlap(owin, iwin)
				if dlength != iwin.length: raise ValueError
			except (TypeError, ValueError):
				raise ValueError('TGC curve does not encompass data window for channel %d' % (rxc,))
			data = data * tgc[np.newaxis,ostart:ostart+dlength]

		# Remove a signal bias, if appropriate
		if debias and iwin.length:
			data -= np.mean(data, axis=1)[:,np.newaxis]

		# Clear the data array
		b[:,:] = 0.
		ws, we = hdr.win.start, hdr.win.end

		if dofht and iwin.length:
			# Ensure that the FHT axis is contiguous for performance
			data = np.asarray(data, order='F')

			# Perform the grouped Hadamard transforms
			for grp, idxmap in sorted(fhts.iteritems()):
				rows = [wset.tx2row(i[1]) for i in idxmap]
				b[rows,ws:we] = fht.fht(data[rows,:], axes=0) / np.sqrt(gsize)
				if signs is not None:
					# Include the sign flips
					b[rows,ws:we] *= signs[:,np.newaxis]
		else:
			# With no FHT, just copy the data
			b[:,ws:we] = data

		if dofft:
			fwdfft()

			# Suppress content out of the band
			c[:,:fswin.start] = 0.
			c[:,fswin.end:] = 0.

			# Bandpass filter the spectral samples
			if len(tails) > 0:
				ltails = len(tails) / 2
				c[:,fswin.start:fswin.start+ltails] *= tails[np.newaxis,:ltails]
				c[:,fswin.end-ltails:fswin.end] *= tails[np.newaxis,-ltails:]

			# Revert to time-domain representation if necessary
			if tdout: invfft()

		# Store the output record in a WaveformSet file
		if tdout:
			hdr = hdr.copy(txgrp=None)
			oset.setrecord(hdr, b[outrows,ws:we], copy=True)
		else:
			hdr = hdr.copy(win=fswin, txgrp=None)
			oset.setrecord(hdr, c[outrows,fswin.start:fswin.end], copy=True)

	# Ensure the output header has been written
	event.wait()

	if not binfile:
		# Write local records to the output WaveformSet
		with lock: oset.store(outfile, append=True)
	else:
		# Map receive channels to rows (slabs) in the output
		rows = dict((i, j) for (j, i) in enumerate(sorted(wset.rxidx)))
		outbin = mio.Slicer(outfile)
		for rxc in wset.rxidx[start::stride]:
			if tdout: b = oset.getrecord(rxc, window=(0, nsamp))[1]
			else: b = oset.getrecord(rxc, window=fswin)[1]
			with lock: outbin[rows[rxc]] = b.T


if __name__ == '__main__':
	grpmap, outpath = None, None
	nprocs = process.preferred_process_count()

	optlist, args = getopt.getopt(sys.argv[1:], 'htp:f:n:o:bg:s:w:m:l:z')

	# Don't populate default options
	kwargs = {}

	for opt in optlist:
		if opt[0] == '-h':
			kwargs['dofht'] = False
		elif opt[0] == '-t':
			kwargs['tdout'] = True
		elif opt[0] == '-p':
			nprocs = int(opt[1])
		elif opt[0] == '-f':
			kwargs['freqs'] = tuple((int(s) if len(s) else None) for s in opt[1].split(':'))
		elif opt[0] == '-n':
			kwargs['nsamp'] = int(opt[1])
		elif opt[0] == '-o':
			outpath = opt[1]
		elif opt[0] == '-b':
			kwargs['binfile'] = True
		elif opt[0] == '-g':
			grpmap = loadkeymat(opt[1], dtype=np.uint32)
		elif opt[0] == '-s':
			kwargs['signs'] = np.loadtxt(opt[1], dtype=bool)
		elif opt[0] == '-w':
			kwargs['window'] = tuple(int(s) for s in opt[1].split(':'))
		elif opt[0] == '-l':
			kwargs['tgcsamps'] = int(opt[1])
		elif opt[0] == '-m':
			kwargs['tgcmap'] = np.loadtxt(opt[1], dtype=np.float32)
		elif opt[0] == '-z':
			kwargs['debias'] = True
		else:
			usage(fatal=True)

	try: infiles = matchfiles(args)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(fatal=True)

	# Determine the propr file extension
	outext = 'fhfft' + ((kwargs.get('binfile', False) and '.mat') or '.wset')

	try:
		outfiles = buildpaths(infiles, outpath, outext)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(fatal=True)

	for infile, outfile in zip(infiles, outfiles):
		print 'Processing data file', infile, '->', outfile
		mpfhfft(nprocs, infile, outfile, grpmap, **kwargs)
