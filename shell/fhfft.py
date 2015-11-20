#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, os, sys, fht, pyfftw, getopt

from collections import defaultdict, OrderedDict

import multiprocessing

from habis.habiconf import matchfiles, buildpaths
from habis.formats import WaveformSet
from pycwp import process, mio

def specwin(nsamp, freqs=None):
	from habis.sigtools import Window

	# Ensure default None is encapsulated
	if freqs is None: freqs = (None,)

	# Find the spectral window
	fs, fe, step = slice(*freqs).indices(nsamp)
	if step != 1:
		raise ValueError('Frequency range must specify consecutive values')

	return Window(fs, end=fe)


def usage(progname=None, fatal=False):
	if progname is None: progname = sys.argv[0]
	binfile = os.path.basename(progname)
	print >> sys.stderr, 'USAGE: %s [-n n] [-h] [-t] [-p p] [-b] [-f s:e] [-s s] [-o outpath] -g g <measurements>' % binfile
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
  -t: Suppress Fourier transform
  -f: Retain only FFT frequency bins in range(s,e) (default: all)
  -n: Override acquisition window to n samples in WaveformSet files
  -o: Provide a path for placing output files
  -b: Write output as a simple 3-D matrix rather than a WaveformSet file
  -s: Correct the decoded signs with the given sign map (default: skip)
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

	Outputs will have the transmission indices rearranged according to the
	rank-2 sequence grpmap, where output transmission i is given the label
	grpmap[i][0] and corresponds to input transmission
	
		j = grpmap[i][1] + grpmap[i][2] * wset.txgrps[1] 
	    
	for wset = WaveformSet.fromfile(infile).
	
	The kwargs contain optional values or default overrides:

	* freqs (default: None): When not None, a sequence (start, end) to be
	  passed as slice(start, end) to record only a frequency of interest.

	* nsamp (default: None): When not None, the nsamp property of the input
	  WaveformSet is forced to this value prior to processing.

	* dofht (default: True): Set to False to disable Hadamard decoding.

	* dofft (default: True): Set to False to disable Fourier transforms.

	* binfile (default: False): Set to True to produce binary matrix
	  output instead of WaveformSet files.

	* signs (default: None): When not None, should be a sequence of length
	  wset.txgrps[1] that specifies a 1 for any local Hadamard index
	  (corresponding to lines in the file) that should be negated, and 0
	  anywhere else. Ignored when dofht is False.

	* start (default: 0) and stride (default: 1): For an input WaveformSet
	  wset, process receive channels in wset.rxidx[start::stride].

	* lock (default: None): If not None, lock.acquire() will be called to
	  serialize writes to output.

	* event (default: None): If not None, event.set() and event.wait() are
	  called to ensure the output WaveformSet header is written before
	  records are appended. The value event.is_set() should be False prior
	  to execution.
	'''
	# Override acquisition window, if desired
	nsamp = kwargs.pop('nsamp', None)

	# Grab synchronization mechanisms
	lock = kwargs.pop('lock', None)
	event = kwargs.pop('event', None)

	# Grab FFT and FHT switches and options
	dofht = kwargs.pop('dofht', True)
	dofft = kwargs.pop('dofft', True)
	freqs = kwargs.pop('freqs', None)

	# Grab output controls
	binfile = kwargs.pop('binfile', False)

	# Grab striding information
	start = kwargs.pop('start', 0)
	stride = kwargs.pop('stride', 1)

	# Grab sign map information
	signs = kwargs.pop('signs', None)

	if len(kwargs):
		raise TypeError("Unrecognized keyword argument '%s'" % kwargs.iterkeys().next())

	# Open the input and create a corresponding output
	wset = WaveformSet.fromfile(infile)

	if nsamp is not None: wset.nsamp = nsamp
	else: nsamp = wset.nsamp

	ntx = wset.ntx

	try:
		gcount, gsize = wset.txgrps
	except TypeError:
		raise ValueError('A valid Tx-group configuration must be specified')

	# Map global indices to transmission number
	outidx = OrderedDict((i[0], i[1] + gsize * i[2]) for i in grpmap)

	# Create a WaveformSet object to hold the ungrouped data
	otype = dofft and _r2c_datatype(wset.dtype) or wset.dtype
	oset = WaveformSet(outidx.keys(), nsamp, wset.f2c, otype)

	# Prepare a list of input rows to be copied to output
	outrows = [wset.tx2row(i) for i in outidx.itervalues()]

	if dofht:
		if not fht.ispow2(gsize):
			raise ValueError('Hadamard transform length must be a power of 2')

		if signs is not None:
			# Ensure signs has values 0 or 1 in the right type
			signs = np.asarray([1 - 2 * s for s in signs], dtype=wset.dtype)
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
			pyfftw.simd_alignment, wset.dtype, order='C')

	if dofft:
		cdim = (ntx, nsamp // 2 + 1)
		c = pyfftw.n_byte_align_empty(cdim,
				pyfftw.simd_alignment, otype, order='C')

		# Find the spectral window
		fswin = specwin(cdim[1], freqs)

		# Create an FFT plan before populating results
		fwdfft = pyfftw.FFTW(b, c, axes=(1,))

	# Create the input file header, if necessary
	getattr(lock, 'acquire', lambda : None)()

	if not getattr(event, 'is_set', lambda : False)():
		if binfile:
			# Create a sliced binary matrix output
			windim = (fswin.length if dofft else nsamp, oset.ntx, wset.nrx)
			mio.Slicer(outfile, dtype=otype, trunc=True, dim=windim)
		else:
			# Create a WaveformSet output
			oset.store(outfile)
		getattr(event, 'set', lambda : None)()

	getattr(lock, 'release', lambda : None)()

	for rxc in wset.rxidx[start::stride]:
		# Grab the waveform record
		hdr, data = wset.getrecord(rxc)

		# Clear the data array
		b[:,:] = 0.
		ws, we = hdr.win.start, hdr.win.end

		if dofht:
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

		# Perform the FFT if desired
		if dofft: fwdfft()

		# Store the output record in a WaveformSet file
		if dofft:
			hdr = hdr.copy(win=fswin, txgrp=None)
			oset.setrecord(hdr, c[outrows,fswin.start:fswin.end], copy=True)
		else:
			hdr = hdr.copy(txgrp=None)
			oset.setrecord(hdr, b[outrows,ws:we], copy=True)

	if not binfile:
		# Write local records to the output WaveformSet
		getattr(event, 'wait', lambda : None)()
		getattr(lock, 'acquire', lambda : None)()
		oset.store(outfile, append=True)
		getattr(lock, 'release', lambda : None)()
	else:
		# Map receive channels to rows (slabs) in the output
		rows = dict((i, j) for (j, i) in enumerate(sorted(wset.rxidx)))
		outbin = mio.Slicer(outfile)
		for rxc in wset.rxidx:
			if dofft: b = oset.getrecord(rxc, window=fswin)[1]
			else: b = oset.getrecord(rxc, window=(0, nsamp))[1]
			getattr(lock, 'acquire', lambda : None)()
			outbin[rows[rxc]] = b.T
			getattr(lock, 'release', lambda : None)()


if __name__ == '__main__':
	# Set default options
	dofht, dofft, binfile = True, True, False
	grpmap, signs, freqs, nsamp, outpath = None, None, None, None, None
	nprocs = process.preferred_process_count()

	optlist, args = getopt.getopt(sys.argv[1:], 'htp:f:n:o:bg:s:')

	for opt in optlist:
		if opt[0] == '-h':
			dofht = False
		elif opt[0] == '-t':
			dofft = False
		elif opt[0] == '-p':
			nprocs = int(opt[1])
		elif opt[0] == '-f':
			freqs = tuple((int(s, base=10) if len(s) else None) for s in opt[1].split(':'))
		elif opt[0] == '-n':
			nsamp = int(opt[1])
		elif opt[0] == '-o':
			outpath = opt[1]
		elif opt[0] == '-b':
			binfile = True
		elif opt[0] == '-g':
			grpmap = np.loadtxt(opt[1], dtype=np.uint32)
		elif opt[0] == '-s':
			signs = np.loadtxt(opt[1], dtype=bool)
		else:
			usage(fatal=True)

	try: infiles = matchfiles(args)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(fatal=True)

	try:
		outfiles = buildpaths(infiles, outpath, 
				'fhfft' + ((binfile and '.mat') or '.wset'))
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(fatal=True)

	for infile, outfile in zip(infiles, outfiles):
		print 'Processing data file', infile, '->', outfile
		mpfhfft(nprocs, infile, outfile, grpmap,
				freqs=freqs, nsamp=nsamp, signs=signs,
				dofht=dofht, dofft=dofft, binfile=binfile)
