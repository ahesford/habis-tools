#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, os, sys, pyfftw, getopt

from fwht import fwht

from collections import defaultdict, OrderedDict
from functools import partial, reduce

import argparse

import multiprocessing

from habis.habiconf import matchfiles, buildpaths
from habis.formats import WaveformSet, loadkeymat, ArgparseLoader
from habis.sigtools import Window, WaveformMap, Waveform
from pycwp import process, mio, cutil

def specwin(nsamp, freqs=None):
	# Ensure default None is encapsulated
	if freqs is None: freqs = (None,)

	# Find the spectral window
	fs, fe, step = slice(*freqs[:2]).indices(nsamp)
	if step != 1:
		raise ValueError('Frequency range must specify consecutive values')

	return Window(fs, end=fe, nonneg=True)


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


def fhfft(infile, outfile, **kwargs):
	'''
	For a real WaveformSet file infile, perform Hadamard decoding and then
	a DFT of the temporal samples. The Hadamard decoding follows the
	grouping configuration stored in the file. The resulting transformed
	records will be stored in the output outfile. The nature of outfile
	depends on the optional argument trmap (see below).

	If trmap is not provided, all records will be written as a binary blob;
	the outfile should be a single string providing the location of the
	output. The output will have shape Ns x Nt x Nr, where Ns is the number
	of output samples per waveform (as governed by the spectral or temporal
	windows applied), Nt is the number of input transmit channels, and Nr
	is the number of input receive channels.

	If trmap is provided, outfile should be a one-to-one map from the keys
	of trmap to output files. A WaveformMap object will be created for each
	key in trmap and stored at the location indicated by the corresponding
	value in outfile.

	Output file(s) will be created or truncated.

	Any TGC parameters in the input, accessible as wset.context['tgc'],
	will be used to adjust the amplitudes of the waveforms prior to
	applying Hadamard and Fourier transforms.

	The kwargs contain optional values or default overrides:

	* grpmap (default: None): When not None, a mapping from output
	  transmit index i and input transmit index

	    j = grpmap[i][0] + grpmap[i][1] * wset.txgrps.size

	  that will be applied to each input WaveformSet wset.

	* freqs (default: None): When not None, a sequence (start, end)
	  to be passed as slice(start, end) to bandpass filter the input after
	  Hadamard decoding.

	* rolloff (default: None): When not None, an integer that defines the
	  half-width of a Hann window that rolls off the bandpass filter
	  specified in freqs. Ignored if freqs is not provided.

	* nsamp (default: None): The length of the time window over which
	  waveforms are considered (and DFTs are performed), starting from
	  global time 0 (i.e., without consideration for input F2C). If None,
	  the value of nsamp in the input is used.

	  ** NOTE: Because the time window always starts at global time 0,
	  a waveform with a data window (start, length) will be cropped when
	  (f2c + start + length) > nsamp, even if nsamp is the value encoded in
	  the file.

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

	* tdout (default: False): Set to True to output time-domain waveforms
	  rather than spectral samples. Preserves input acquisition windows.

	* signs (default: None): When not None, should be a sequence of length
	  wset.txgrps.size that specifies a 1 for any local Hadamard index
	  (corresponding to lines in the file) that should be negated, and 0
	  anywhere else. Ignored when an FHT is not performed.

	* trmap (default: None): If provided, must be a map from a label
	  (referencing an output location in the map outfile) to a map from
	  receive indices to lists of transmit indices that, together, identify
	  transmit-receive pairs to extract from the input.

	* start (default: 0) and stride (default: 1): For an input WaveformSet
	  wset, process receive channels in wset.rxidx[start::stride].

	* lock (default: None): If not None, it should be a context manager
	  that is invoked to serialize writes to output.

	* event (default: None): Only used then trmap is not provided. If not
	  None, event.set() and event.wait() are called to ensure the output
	  header is written to the binary-blob output before records are
	  appended. The value event.is_set() should be False prior to
	  execution.
	'''
	# Override acquisition window, if desired
	nsamp = kwargs.pop('nsamp', None)

	# Grab synchronization mechanisms
	try: lock = kwargs.pop('lock')
	except KeyError: lock = multiprocessing.Lock()
	try: event = kwargs.pop('event')
	except KeyError: event = multiprocessing.Event()

	# Grab FFT and FHT switches and options
	tdout = kwargs.pop('tdout', False)
	freqs = kwargs.pop('freqs', None)
	rolloff = kwargs.pop('rolloff', None)
	dofft = (freqs is not None) or not tdout

	if freqs is not None:
		flo, fhi = freqs
		if rolloff and not 0 < rolloff < (fhi - flo) // 2:
			raise ValueError('Rolloff must be None or less than half bandwidth')

	# Grab striding information
	start = kwargs.pop('start', 0)
	stride = kwargs.pop('stride', 1)

	# Grab sign map information
	signs = kwargs.pop('signs', None)

	# Grab the number of samples per TGC value and an optional gain map
	tgcsamps = kwargs.pop('tgcsamps', None)
	tgcmap = kwargs.pop('tgcmap', None)

	trmap = kwargs.pop('trmap', None)

	grpmap = kwargs.pop('groupmap', None)

	if len(kwargs):
		raise TypeError(f"Unrecognized keyword '{next(iter(kwargs))}'")

	# Open the input and create a corresponding output
	wset = WaveformSet.load(infile)

	# Pull default sample count from input file
	if nsamp is None: nsamp = wset.nsamp
	elif wset.nsamp < nsamp: wset.nsamp = nsamp

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

	# Figure out the data type of compensated waveforms
	if len(tgc): itype = np.dtype(wset.dtype.type(0) * tgc.dtype.type(0))
	else: itype = wset.dtype

	# Create a WaveformSet object to hold the ungrouped data
	ftype = _r2c_datatype(itype)
	otype = ftype if not tdout else itype

	if grpmap is not None:
		# Make sure the WaveformSet has a local configuration
		try:
			gcount, gsize = wset.txgrps
		except TypeError:
			raise ValueError('A valid Tx-group configuration is required')

		if gsize < 1 or (gsize & (gsize - 1)):
			raise ValueError('Hadamard length must be a positive power of 2')

		# Validate local portion of the group map and assign
		wset.groupmap = grpmap

		if signs is not None:
			# Ensure signs has values 0 or 1 in the right type
			signs = np.asarray([1 - 2 * s for s in signs], dtype=itype)
			if signs.ndim != 1 or len(signs) != gsize:
				msg = f'Sign list must have shape ({wset.txgrps[1]},)'
				raise ValueError(msg)

		# Map group configurations back to element indices
		invgroups = { (li, g): i for i, (li, g) in wset.groupmap.items() }

		# Identify all FHTs represented in the input transmissions
		fhts = defaultdict(list)
		for i in wset.txidx:
			gi, li = i // gsize, i % gsize
			# Find element index for this decoded transmission slot
			el = invgroups[li, gi]
			fhts[gi].append((li, el))

		# Enforce ordering of the FHT groups
		fhts = OrderedDict(sorted(fhts.items()))

		# Sort indices in Hadamard index order and verify groups
		# Also map an element index to a global row index
		el2row = { }
		k = 0
		for i, v in fhts.items():
			if len(v) != gsize:
				raise ValueError('Incomplete Hadamard group {i}')
			v.sort()
			for j, vl in enumerate(v):
				if j != vl[0]:
					msg = f'Invalid local index in Hadamard group {i}'
					raise ValueError(msg)
				el2row[vl[1]] = k
				k += 1
	else:
		if wset.txgrps is not None:
			raise ValueError('A groupmap is required when input includes Tx-group configuration')
		# Dummy map of elements to rows
		el2row = {i: j for j, i in enumerate(wset.txidx)}

	# Create intermediate (FHT) and output (FHFFT) arrays
	# FFT axis is contiguous for FFT performance
	b = pyfftw.empty_aligned((wset.ntx, nsamp), dtype=itype, order='C')

	if dofft:
		# Create FFT output and a plan
		cdim = (wset.ntx, nsamp // 2 + 1)
		c = pyfftw.empty_aligned(cdim, dtype=ftype, order='C')
		fwdfft = pyfftw.FFTW(b, c, axes=(1,), direction='FFTW_FORWARD')

		# Create an inverse FFT plan for time-domain output
		if tdout:
			invfft = pyfftw.FFTW(c, b, axes=(1,), direction='FFTW_BACKWARD')

		# Find the spectral window of interest
		fswin = specwin(cdim[1], freqs)

		# Try to build bandpass tails
		if rolloff: tails = np.hanning(2 * int(rolloff))
		else: tails = np.array([])

	if trmap:
		# Identify the subset of receive channels needed
		allrx = reduce(set.union, (trm.keys() for trm in trmap.values()), set())
		rxneeded = sorted(allrx.intersection(wset.rxidx))[start::stride]
	else: 
		rxneeded = wset.rxidx[start::stride]

		# In blob mode, the first write must create a header
		with lock:
			if not event.is_set():
				# Create a sliced binary matrix output
				windim = (nsamp if tdout else fswin.length, wset.ntx, wset.nrx)
				mio.Slicer(outfile, dtype=otype, trunc=True, dim=windim)
				event.set()

		# Ensure the output header has been written
		event.wait()

		# Map receive channels to rows (slabs) in the output
		row2slab = dict((i, j) for (j, i) in enumerate(sorted(wset.rxidx)))
		# Map transmit channels to decoded FHT rows
		outrows = [r for (e,r) in sorted(el2row.items())]

		outbin = mio.Slicer(outfile)

	for rxc in rxneeded:
		# Find the input window relative to 0 f2c
		iwin = wset.getheader(rxc).win.shift(wset.f2c)
		owin = (0, nsamp)

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

		# Read the data over the input window
		data = wset.getrecord(rxc, window=iwin)[1]

		# Time-gain compensation
		if len(tgc) and iwin.length:
			twin = (0, len(tgc))
			try:
				ostart, istart, dlength = cutil.overlap(twin, iwin)
				if dlength != iwin.length: raise ValueError
			except (TypeError, ValueError):
				raise ValueError('TGC curve does not encompass data window for channel %d' % (rxc,))
			data = data * tgc[np.newaxis,ostart:ostart+dlength]

		# Clear the data array
		b[:,:] = 0.
		ws, we = owin.start, owin.end

		if wset.groupmap and iwin.length:
			# Ensure that the FHT axis is contiguous for performance
			data = np.asarray(data, order='F')

			# Perform the grouped Hadamard transforms
			k = 0
			for grp, idxmap in fhts.items():
				# Input rows require tranmsit-to-row mapping
				irows = [wset.tx2row(grp * gsize + i)
							for i in range(gsize)]

				# Perform the decode if necessary
				if gsize > 1 and iwin.length:
					d = fwht(data[irows,:], axes=0) / np.sqrt(gsize)
					b[k:k+gsize,ws:we] = d
				else: b[k:k+gsize,ws:we] = data[irows,:]

				# Include the sign flips if necessary
				if signs is not None:
					b[k:k+gsize,ws:we] *= signs[:,np.newaxis]
				k += gsize
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
				ltails = len(tails) // 2
				c[:,fswin.start:fswin.start+ltails] *= tails[np.newaxis,:ltails]
				c[:,fswin.end-ltails:fswin.end] *= tails[np.newaxis,-ltails:]

			# Revert to time-domain representation if necessary
			if tdout: invfft()

		if not trmap:
			# Write the binary blob for this receive channel
			orow = row2slab[rxc]
			with lock:
				if tdout: outbin[orow] = b[outrows,:].T
				else: outbin[orow] = c[outrows,fswin.start:fswin.end].T
			# Nothing more to do in blob mode
			continue

		# Slice desired range from output data
		if tdout:
			dblock = b[:,ws:we]
			dstart = ws
		else:
			dblock = c[:,fswin.start:fswin.end]
			dstart = fswin.start

		for label, trm in trmap.items():
			# Pull tx list for this tier and rx channel, if possible
			try: tl = trm[rxc]
			except KeyError: tl = [ ]

			if not len(tl): continue

			# Collect all transmissions for this rx channel
			wmap = WaveformMap()
			for t in tl:
				# Make sure transmission is represented in output
				try: row = el2row[t]
				except KeyError: continue

				wave = Waveform(nsamp, dblock[row], dstart)
				wmap[t, rxc] = wave

			# Flush the waveform map to disk
			with lock: wmap.store(outfile[label], append=True)


def in2out(infile, outpath, labels=None):
	'''
	Map a single input file into one or more output files.

	If labels is None,

	  habis.habiconf.buildpaths([infile], outpath, 'fhfft.mat')[0]

	is invoked and returned to produce a single output file.

	Otherwise, buildpaths is called repeatedly as above, with the extension
	'fhfft.mat' generally replaced by f'{l}.wmz' for each element l in the
	collection or iterator labls. The output will be a dictionary mapping
	each element of labels to the corresponding buildpaths output. As a
	special case, if l is the empty string for any element in labels, the
	extension will just be 'wmz' (i.e., the extension will be constructed
	to avoid two dots with no interceding character).

	This will pass along any errors raised by IOError or attempts to
	iterate through labels.
	'''
	if not labels:
		return buildpaths([infile], outpath, 'fhfft.mat')[0]

	return { l: buildpaths([infile], outpath,
				l and f'{l}.wmz' or 'wmz')[0] for l in labels }


if __name__ == '__main__':
	nptxtloader = partial(ArgparseLoader, np.loadtxt)
	ikmloader = partial(ArgparseLoader, loadkeymat, scalar=False, dtype=np.uint32)

	parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
			description='Filter, decode and descramble WaveformSet files')

	parser.add_argument('-p', '--procs', type=int,
			default=process.preferred_process_count(),
			help='Use PROCS processes in parallel')

	parser.add_argument('-t', '--tdout', action='store_true',
			help='Produce time-domain, not spectral, output')

	parser.add_argument('-f', '--freqs', metavar=('START', 'END'), type=int,
			nargs=2, help='Spectral bandwidth of output in DFT bins')

	parser.add_argument('-r', '--rolloff', type=int,
			help='Frequency rolloff of output in DFT bins')

	parser.add_argument('-n', '--nsamp', type=int,
			help='Override length of acquisition window')

	parser.add_argument('-o', '--outpath', default=None,
			help='Store output in OUTPATH (default: alongside input)')

	parser.add_argument('-s', '--signs', type=nptxtloader(dtype=bool),
			help='List of signs applied before Hadamard decoding')

	parser.add_argument('-l', '--tgc-length', dest='tgcsamps', type=int,
			help='Number of TGC samples per TGC value in a WaveformSet')

	parser.add_argument('-m', '--tgc-map',
			dest='tgcmap', type=nptxtloader(dtype='float32'),
			help='Two-column file mapping nominal to actual gain')

	parser.add_argument('-g', '--groupmap', type=ikmloader(),
			help='Global transmit groupmap to assign to input files')

	parser.add_argument('-T', '--trmap', action='append', type=ikmloader(),
			help='T-R map of measurement pairs to store (multiples OK)')

	parser.add_argument('-L', '--trlabel', action='append',
			help='Label for provided TR map (one per -T flag)')

	parser.add_argument('input', nargs='+', help='List of input files to process')

	args = parser.parse_args(sys.argv[1:])

	# Special case: handle the T-R maps
	trmap = getattr(args, 'trmap', [])
	try:
		trlab = args.trlabel
		delattr(args, 'trlabel')
	except AttributeError: trlab = [ ]
	if len(trmap) != len(trlab):
		sys.exit(f'ERROR: must specify same number of -L and -T arguments')
	if len(trlab) != len(set(trlab)):
		sys.exit(f'ERROR: all labels provided with -L must be unique')
	args.trmap = dict(zip(trlab, trmap))

	# Convert args namespace to kwargs
	kwargs = { }

	# Build the keyword arguments
	attrs = { d for d in dir(args) if not d.startswith('_') }
	for attr in attrs.difference({'procs', 'outpath', 'input'}):
		kwargs[attr] = getattr(args, attr)

	try: infiles = matchfiles(args.input)
	except IOError as e: sys.exit(f'ERROR: {e}')

	for infile in infiles:
		print('Processing data file', infile)
		try: outfile = in2out(infile, args.outpath, args.trmap.keys())
		except IOError as e: sys.ext(f'ERROR: {e}')
		mpfhfft(args.procs, infile, outfile, **kwargs)
