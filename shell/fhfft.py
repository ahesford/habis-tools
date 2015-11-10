#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, os, sys, fht, pyfftw, getopt, operator as op
from collections import defaultdict

import multiprocessing

from habis.formats import WaveformSet
from pycwp import process


def usage(progname=None):
	if progname is None: progname = sys.argv[0]
	binfile = os.path.basename(progname)
	print >> sys.stderr, 'USAGE: %s [-l l] [-p p] [-f start:end] <measurements> <outfile>' % binfile
	print >> sys.stderr, '''
  Preprocess HABIS measurement data by Hadamard decoding transmissions and
  Fourier transforming the time-domain data. Measurement data is contained in
  the 'measurements' WaveformSet file.

  The processed data is transposed between nodes so that, on output, each node
  contains all channel measurements for a subset of the transmissions included
  in each measurement input. The output data is saved to files with a name
  format given in outfmt, which is used to generate output files named
  outfmt.format(tidx) for each integer transmission index tidx.

  OPTIONAL ARGUMENTS:
  -p: Use p processors (default: all available processors)
  -l: Assume Hadamard coding in groups of l elements (default: 2048)
  -f: Retain only FFT frequency bins in range(start, end) (default: all)
	'''

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


def mpfhfft(infile, lfht, outfile, nproc, freqrange=(None,)):
	'''
	Subdivide, along receive channels, the work of fhfft() among nproc
	processes to Hadamard-decode and Fourier transform the WaveformSet
	stored in infile into a WaveformSet file that will be written to
	outfile. The output file will be overwritten if it already exists.
	'''
	# Copy the input header and get the receive-channel indices
	wset = WaveformSet.fromfile(infile)
	rxidx = wset.rxidx
	# Change the data type of the output
	odtype = _r2c_datatype(wset.dtype)[1]
	open(outfile, 'wb').write(wset.encodefilehdr(dtype=odtype))

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
			args = (infile, lfht, rxidx[i::nproc], outfile, freqrange, lock)
			pool.addtask(target=fhfft, name=procname, args=args)
		pool.start()
		pool.wait()


def fhfft(infile, lfht, rxchans, outfile, freqrange=(None,), lock=None):
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
	oset = WaveformSet.empty_like(wset)

	# Set the right input and output types for the transformed data
	itype, otype = _r2c_datatype(oset.dtype)
	oset.dtype = otype

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
	sl = slice(*freqrange)
	fs, fe, step = sl.indices(cdim[1])
	if step != 1:
		raise ValueError('Frequency range must specify consecutive values')
	
	# Create an FFT plan before populating results
	fwdfft = pyfftw.FFTW(b, c, axes=(1,))

	for rxc in rxchans:
		# Grab the waveform record
		hdr, data = wset.getrecord(rxc)
		
		# Ensure that the FHT axis is contiguous for performance
		data = np.asarray(data, order='F')

		# Clear the data array
		b[:,:] = 0.
		ws, wl = hdr['win']
		we = ws + wl

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
		hdr['win'][:] = fs, (fe - fs)
		oset.setrecord(hdr, c[:,fs:fe], copy=True)

	# Write local records to the output
	try: lock.acquire()
	except AttributeError: pass

	oset.store(outfile, append=True)

	try: lock.release()
	except AttributeError: pass


if __name__ == '__main__':
	# Set default options
	lfht, freqrange = 2048, (None,)
	nprocs = process.preferred_process_count()

	optlist, args = getopt.getopt(sys.argv[1:], 'hl:p:f:')

	for opt in optlist:
		if opt[0] == '-l':
			lfht = int(opt[1])
		elif opt[0] == '-p':
			nprocs = int(opt[1])
		elif opt[0] == '-f':
			freqrange = tuple((int(s, base=10) if len(s) else None) for s in opt[1].split(':'))
		else:
			usage()
			sys.exit(1)

	if len(args) != 2:
		usage()
		sys.exit()

	infile, outfile = args

	print 'Processing data file', infile, 'with coding length', lfht
	mpfhfft(infile, lfht, outfile, nprocs, freqrange)
