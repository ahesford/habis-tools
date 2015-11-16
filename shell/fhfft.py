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
	print >> sys.stderr, 'USAGE: %s [-n n] [-l l] [-p p] [-b] [-r r] [-f start:end] [-o output] <measurements>' % binfile
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
  -l: Assume Hadamard coding in groups of l elements (default: 2048)
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
		return np.dtype('float32'), np.dtype('complex64')
	else:
		return np.dtype('float64'), np.dtype('complex128')


def mpfhfft(infile, lfht, outfile, nproc, freqrange=(None,), 
		nsamp=None, binfile=False, rdl=None):
	'''
	Subdivide, along receive channels, the work of fhfft() among nproc
	processes to Hadamard-decode and Fourier transform the WaveformSet
	stored in infile into a WaveformSet file that will be written to
	outfile. The output file will be overwritten if it already exists.

	If nsamp is not None, the acquisition window of the input and output
	WaveformSet files will be forced to the specified value.
	'''
	# Copy the input header and get the receive-channel indices
	wset = WaveformSet.fromfile(infile)
	if nsamp is not None: wset.nsamp = nsamp

	rxidx = wset.rxidx
	# Change the data type of the output
	odtype = _r2c_datatype(wset.dtype)[1]

	if not binfile:
		# Start a new WaveformSet file
		open(outfile, 'wb').write(wset.encodefilehdr(dtype=odtype))
	else:
		# Determine the spectral window to write
		fswin = specwin(freqrange, wset.nsamp // 2 + 1)
		# Create a sliced binary matrix
		mio.Slicer(outfile, dtype=odtype, trunc=True,
				dim=(fswin.length, wset.ntx, wset.nrx))

	# Allow the memory-mapped input to be closed
	del wset

	# Create a multiprocessing lock to serialize output access
	lock = multiprocessing.Lock()

	# Span the desired processes to perform FHFFT
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Give each process a meaningful name
			procname = process.procname(i)
			# Stride the receive channels
			args = (infile, lfht, rxidx[i::nproc], outfile,
					freqrange, nsamp, lock, binfile, rdl)
			pool.addtask(target=fhfft, name=procname, args=args)
		pool.start()
		pool.wait()


def fhfft(infile, lfht, rxchans, outfile, freqrange=(None,), 
		nsamp=None, lock=None, binfile=False, rdl=None):
	'''
	For a real WaveformSet file infile, tile a series of Hadamard
	transforms of length lfht along the transmit-index dimension of each
	record in rxchans, and then perform a DFT of the temporal samples. The
	resulting transformed records will be stored in the output file
	outfile, which should already exist and contain a compatible
	WaveformSet file header. If lock is provided, it will be acquired and
	released using lock.acquire() and lock.release(), respectively,
	immediately prior to and following the append of transformed records to
	outfile.
	
	If freqrange is specified, it should be a sequence (start, end), to be
	passed as the first two argument to slice() to extract and store
	frequencies of interest from the transformed temporal samples in each
	record. A third (step) argument is not supported.

	The length of the Hadamard transforms must be a power of 2, and the
	number of transmit indices must be an integer multiple of the Hadamard
	transform length.
	'''
	if not fht.ispow2(lfht):
		raise ValueError('Hadamard transform length must be a power of 2')

	# Open the input and create a corresponding output
	wset = WaveformSet.fromfile(infile)
	if nsamp is not None: wset.nsamp = nsamp

	# Set the right input and output types for the transformed data
	itype, otype = _r2c_datatype(wset.dtype)

	# Prepare output storage
	if not binfile:
		oset = WaveformSet.empty_like(wset)
		oset.dtype = otype
	else:
		oset = mio.Slicer(outfile)

	# Determine the transform sizes
	nfft, nfht = wset.nsamp, wset.ntx

	if nfht % lfht != 0:
		raise ValueError('Transmit index cound must be integer multiple of FHT length')

	# Create intermediate (FHT) and output (FHFFT) arrays
	# FFT axis is contiguous for FFT performance
	b = pyfftw.n_byte_align_empty((nfht, nfft), pyfftw.simd_alignment, itype, order='C')
	cdim = (nfht, nfft // 2 + 1)
	c = pyfftw.n_byte_align_empty(cdim, pyfftw.simd_alignment, otype, order='C')

	# Find the spectral window
	fswin = specwin(freqrange, cdim[1])
	
	# Create an FFT plan before populating results
	fwdfft = pyfftw.FFTW(b, c, axes=(1,))

	def mutex(f):
		"Call the function f (with no arguments) after acquiring a lock"
		try: lock.acquire()
		except AttributeError: pass
		f()
		try: lock.release()
		except AttributeError: pass


	for rxc in rxchans:
		# Grab the waveform record
		hdr, data = wset.getrecord(rxc)
		
		# Ensure that the FHT axis is contiguous for performance
		data = np.asarray(data, order='F')

		# Clear the data array
		b[:,:] = 0.
		ws, we = hdr.win.start, hdr.win.end

		# Perform the tiled Hadamard transforms
		if lfht > 1:
			for s in range(0, nfht, lfht):
				e = s + lfht
				b[s:e,ws:we] = fht.fht(data[s:e,:], axes=0)
		else:
			# Skip 1-point Hadamard transforms
			b[:,ws:we] = data

		# Perform the FFT
		fwdfft()

		# Record the output record
		hdr = WaveformSet.recordhdr(hdr.idx, hdr.pos, fswin)
		if not binfile:
			oset.setrecord(hdr, c[:,fs:fe], copy=True)
		else:
			idx = rxc if rdl is not None else rdl[rxc]
			mutex(lambda : oset[idx] = c[:,fs:fe].T)

	# Write local records to the output WaveformSet
	if not binfile:
		mutex(lambda : oset.store(outfile, append=True))


if __name__ == '__main__':
	# Set default options
	lfht, freqrange, nsamp, outfiles = 2048, (None,), None, None
	binfile, rdl = False, None
	nprocs = process.preferred_process_count()

	optlist, args = getopt.getopt(sys.argv[1:], 'hl:p:f:n:o:br:')

	for opt in optlist:
		if opt[0] == '-l':
			lfht = int(opt[1])
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

	infiles = []
	for arg in args:
		if os.path.lexists(arg): infiles.append(arg)
		else: infiles.extend(glob.glob(arg))

	if outfiles is None:
		ext = (binfile and '.mat') or '.wset'
		outfiles = [os.path.splitext(f)[0] + '.fhfft' + ext  for f in infiles]

	if len(infiles) < 1:
		print >> sys.stderr, 'ERROR: No input files'
		usage(fatal=True)
	elif len(infiles) != len(outfiles):
		print >> sys.stderr, 'ERROR: output name count disagrees with input name count'
		usage(fatal=True)

	for infile, outfile in zip(infiles, outfiles):
		print 'Processing data file', infile, 'with coding length', lfht
		mpfhfft(infile, lfht, outfile, nprocs, freqrange, nsamp, binfile, rdl)
