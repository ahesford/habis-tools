#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, os, sys, fht, pyfftw, getopt, glob
from collections import defaultdict

import multiprocessing

from habis.formats import WaveformSet
from pycwp import process, mio

def specwin(freqrange, nsamp):
	from habis.sigtools import Window

	# Find the spectral window
	fs, fe, step = slice(*freqrange).indices(nsamp)
	if step != 1:
		raise ValueError('Frequency range must specify consecutive values')

	return Window(fs, end=fe)


def usage(progname=None, fatal=False):
	if progname is None: progname = sys.argv[0]
	binfile = os.path.basename(progname)
	print >> sys.stderr, 'USAGE: %s [-n n] [-h] [-t] [-p p] [-b] [-r r] [-f start:end] [-o output] <measurements>' % binfile
	print >> sys.stderr, '''
  Preprocess HABIS measurement data by Hadamard decoding transmissions and
  Fourier transforming the time-domain data. Measurement data is contained in
  the 'measurements' WaveformSet files. If a filesystem object with the
  specified exact name does not exist, it is treated as a glob to look for files.

  Output is stored, by default, in a WaveformSet file whose name has the same
  name as the input with any extension swapped with '.fhfft.wset'. If the input
  file has no extension, an extension is appended.

  OPTIONAL ARGUMENTS:
  -p: Use p processors (default: all available processors)
  -h: Suppress Hadamard decoding
  -t: Suppress Fourier transform
  -f: Retain only FFT frequency bins in range(start, end) (default: all)
  -n: Override acquisition window to n samples in WaveformSet files
  -o: Override the output file name (valid only for single measurement-file input)
  -b: Write output as a simple 3-D matrix rather than a WaveformSet file
  -r: Read a file r for an ordering of output channels (ignored for WaveformSet files)
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


def mpfhfft(infile, outfile, nproc, freqrange=(None,), nsamp=None,
		binfile=False, rdl=None, dofht=True, dofft=True):
	'''
	Subdivide, along receive channels, the work of fhfft() among nproc
	processes to Hadamard-decode and Fourier transform the WaveformSet
	stored in infile into a WaveformSet file that will be written to
	outfile. The output file will be overwritten if it already exists.

	All arguments except nproc are interpreted as corresponding arguments
	in fhfft().
	'''
	# Copy the input header and get the receive-channel indices
	wset = WaveformSet.fromfile(infile)
	if nsamp is not None: wset.nsamp = nsamp

	rxidx = wset.rxidx

	if binfile:
		if dofft:
			odtype = _r2c_datatype(wset.dtype)
			fswin = specwin(freqrange, wset.nsamp // 2 + 1)
			windim = (fswin.length, wset.ntx, wset.nrx)
		else:
			odtype = wset.dtype
			windim = (wset.nsamp, wset.ntx, wset.nrx)
		# Create a sliced binary matrix
		mio.Slicer(outfile, dtype=odtype, trunc=True, dim=windim)

	# Allow the memory-mapped input to be closed
	del wset

	# Create a multiprocessing lock and event to serialize output access
	lock = multiprocessing.Lock()
	event = multiprocessing.Event()

	# Span the desired processes to perform FHFFT
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Give each process a meaningful name
			procname = process.procname(i)
			# Stride the receive channels
			args = (infile, rxidx[i::nproc], outfile, freqrange,
					nsamp, binfile, rdl, dofht, dofft, lock, event)
			pool.addtask(target=fhfft, name=procname, args=args)
		pool.start()
		pool.wait()


def fhfft(infile, rxchans, outfile, freqrange=(None,), nsamp=None,
		binfile=False, rdl=None, dofht=True, dofft=True, lock=None, event=None):
	'''
	For a real WaveformSet file infile, perform Hadamard decoding and then
	a DFT of the temporal samples. The resulting transformed records will
	be stored in the output file outfile, which will be created or
	truncated. The Hadamard decoding followings the grouping configuration
	stored in the file.

	If freqrange is specified, it should be a sequence (start, end), to be
	passed as the first two argument to slice() to extract and store
	frequencies of interest from the transformed temporal samples in each
	record. A third (step) argument is not supported.

	The length of the Hadamard transforms must be a power of 2.

	If one of dofht or dofft is False, this aspect of the transformations
	will be skipped.

	If lock is provided, it will be acquired and released using
	lock.acquire() and lock.release(), respectively, immediately prior to
	and following the append of transformed records to outfile. If event is
	provided, the event.set() and event.wait() will be used to ensure the
	file header has been written.
	'''
	# Open the input and create a corresponding output
	wset = WaveformSet.fromfile(infile)
	if nsamp is not None: wset.nsamp = nsamp

	if dofht:
		try: gcount, gsize = wset.txgrps
		except TypeError:
			raise ValueError('Hadamard decoding requires valid Tx-group configuration')

		if not fht.ispow2(gsize):
			raise ValueError('Hadamard transform length must be a power of 2')

		# Build a map between transmission indices and (index, group) pairs
		fhts = defaultdict(list)
		for i in wset.txidx:
			gi, li = i // gsize, i % gsize
			fhts[gi].append((li, i))

		# Sort indices in Hadamard index order
		for i, v in fhts.iteritems():
			if len(v) != gsize:
				raise ValueError('Hadamard group %d does not contain full channel set' % i)
			v.sort()

	# Set the right input and output types for the transformed data
	otype = _r2c_datatype(wset.dtype) if dofft else wset.dtype

	# Prepare output storage
	if binfile:
		oset = mio.Slicer(outfile)
	else:
		oset = WaveformSet.empty_like(wset)
		oset.dtype = otype

		# Create the input file header, if necessary
		getattr(lock, 'acquire', lambda : None)()

		if not getattr(event, 'is_set', lambda : False)():
			oset.store(outfile)
			getattr(event, 'set', lambda : None)()

		getattr(lock, 'release', lambda : None)()

	# Determine the transform sizes
	nsamp, ntx = wset.nsamp, wset.ntx

	# Create intermediate (FHT) and output (FHFFT) arrays
	# FFT axis is contiguous for FFT performance
	b = pyfftw.n_byte_align_empty((ntx, nsamp),
			pyfftw.simd_alignment, wset.dtype, order='C')

	if dofft:
		cdim = (ntx, nsamp // 2 + 1)
		c = pyfftw.n_byte_align_empty(cdim,
				pyfftw.simd_alignment, otype, order='C')

		# Find the spectral window
		fswin = specwin(freqrange, cdim[1])

		# Create an FFT plan before populating results
		fwdfft = pyfftw.FFTW(b, c, axes=(1,))

	for rxc in rxchans:
		# Grab the waveform record
		hdr, data = wset.getrecord(rxc)

		# Clear the data array
		b[:,:] = 0.
		ws, we = hdr.win.start, hdr.win.end

		if dofht:
			# Ensure that the FHT axis is contiguous for performance
			data = np.asarray(data, order='F')

			# Perform the grouped Hadamard transforms
			for grp, idxmap in sorted(fhts):
				rows = [wset.tx2row(i[1]) for i in idxmap]
				b[rows,ws:we] = fht.fht(data[rows,:], axes=0)
		else:
			# Skip 1-point Hadamard transforms
			b[:,ws:we] = data

		# Perform the FFT if desired
		if dofft: fwdfft()

		if not binfile:
			# Store the output record in a WaveformSet file
			if dofft:
				hdr = hdr.copy(win=fswin)
				oset.setrecord(hdr, c[:,fswin.start:fswin.end], copy=True)
			else:
				oset.setrecord(hdr, b[:,ws:we], copy=True)
		else:
			# Binary matrix output goes right to disk
			idx = rxc if rdl is None else rdl[rxc]
			getattr(lock, 'acquire', lambda : None)()
			if dofft: oset[idx] = c[:,fswin.start:fswin.end].T
			else: oset[idx] = b.T
			getattr(lock, 'release', lambda : None)()

	if not binfile:
		# Write local records to the output WaveformSet
		getattr(event, 'wait', lambda : None)()
		getattr(lock, 'acquire', lambda : None)()
		oset.store(outfile, append=True)
		getattr(lock, 'release', lambda : None)()


if __name__ == '__main__':
	# Set default options
	dofht, dofft, freqrange, nsamp, outfiles = True, True, (None,), None, None
	binfile, rdl = False, None
	nprocs = process.preferred_process_count()

	optlist, args = getopt.getopt(sys.argv[1:], 'htp:f:n:o:br:')

	for opt in optlist:
		if opt[0] == '-h':
			dofht = False
		elif opt[0] == '-t':
			dofft = False
		elif opt[0] == '-p':
			nprocs = int(opt[1])
		elif opt[0] == '-f':
			freqrange = tuple((int(s, base=10) if len(s) else None) for s in opt[1].split(':'))
		elif opt[0] == '-n':
			nsamp = int(opt[1])
		elif opt[0] == '-o':
			outfiles = [opt[1]]
		elif opt[0] == '-b':
			binfile = True
		elif opt[0] == '-r':
			rdl = np.loadtxt(opt[1])
		else:
			usage(fatal=True)

	if not (dofht or dofft):
		raise ValueError('ERROR: FHT and FFT cannot both be disabled')

	infiles = [f for arg in args for f in glob.glob(arg)]

	if outfiles is None:
		ext = (binfile and '.mat') or '.wset'
		outfiles = [os.path.splitext(f)[0] + '.fhfft' + ext for f in infiles]

	if len(infiles) < 1:
		print >> sys.stderr, 'ERROR: No input files'
		usage(fatal=True)
	elif len(infiles) != len(outfiles):
		print >> sys.stderr, 'ERROR: output name count disagrees with input name count'
		usage(fatal=True)

	for infile, outfile in zip(infiles, outfiles):
		print 'Processing data file', infile
		mpfhfft(infile, outfile, nprocs, freqrange,
				nsamp, binfile, rdl, dofht, dofft)
