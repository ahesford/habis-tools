#!/usr/bin/env python

import numpy as np, os, sys, fht, pyfftw, getopt
from multiprocessing import Process, cpu_count

from pyajh import mio


def usage(progname=None):
	if progname is None: progname = sys.argv[0]
	binfile = os.path.basename(progname)
	print >> sys.stderr, 'USAGE: %s [-l l] [-p p] [-f start:end:skip] [-t tmpdir] <measurements>' % binfile
	print '''
  Preprocess HABIS measurement data by Hadamard decoding transmissions and
  Fourier transforming the time-domain data. Measurement data is listed in the
  'measurements' file which specifies, on each line, the index of a receiver
  channel followed by a path to the file name containing a full set of
  time-domain data for the channel.

  OPTIONAL ARGUMENTS:
  -p: Use p processors (default: all available processors)
  -l: Assume Hadamard coding in groups of l elements (default: 2048)
  -f: Retain only FFT frequency bins in slice(start, end, skip) (default: all)
  -t: Store spectral output files in tmpdir (default: same as measurements)
	'''


def fhfft(infiles, lfht, freqrange=[None], tmpdir=None):
	'''
	For N x M real matrices A stored in the binary files infiles, tile a
	series of Hadamard transforms (each of length lfht) along the M
	dimension, and then perform a DFT along the N dimension to produce an
	output matrix B. Save the matrix B[S,:], where S = slice(*freqrange).
	Each matrix B is saved in a file whose name is determined by appending
	'.spectral' to the corresponding value in infiles.
	
	The length of the Hadamard transforms must be a power of 2, and M must
	be an integer multiple of the Hadamard transform length.
	'''
	if not fht.ispow2(lfht):
		raise ValueError('Hadamard transform length must be a power of 2')

	# Keep a record of the FFT and Hadamard sizes
	nfft, nfht = 0, 0

	for infile in infiles:
		# Read the input file
		# Transpose and copy for faster FHT performance
		a = mio.readbmat(infile).T.copy()
		
		if a.dtype != np.float32 and not a.dtype != np.float64:
			raise TypeError('Matrix must have a dtype of float32 or float64')
		
		if len(a.shape) != 2:
			raise ValueError('Matrix must be two-dimensional')
		
		# If the dimensions have changed, create new arrays for the FFT
		if a.shape != (nfht, nfft):
			# Copy the new planned shape
			nfht, nfft = a.shape
			
			if nfht % lfht != 0:
				raise ValueError('Second matrix dimension must be integer multiple of Hadamard transform length')
			
			# Create an intermediate array to store Hadamard results
			b = pyfftw.n_byte_align_empty((nfht, nfft), pyfftw.simd_alignment, a.dtype)
			
			# Create an output array to store the final FFT results
			# The length of the FFT axis is half the input, plus one for DC
			cdim = (nfht, nfft // 2 + 1)
			cdtype = np.complex64 if (a.dtype == np.float32) else np.complex128
			c = pyfftw.n_byte_align_empty(cdim, pyfftw.simd_alignment, cdtype)
			
			# Create an FFT plan before populating the Hadamard results
			fwdfft = pyfftw.FFTW(b, c, axes=(1,))
		
		# At this point, the gain compensation should be applied
		
		# Perform the tiled Hadamard transforms
		for s in range(0, nfht, lfht):
			e = s + lfht
			b[s:e,:] = fht.fht(a[s:e,:], axes=0)
			
		# At this point, time-gating should be applied
		
		# Perform the FFT
		fwdfft()

		outfile = infile + '.spectral'
		if tmpdir is not None:
			outfile = os.path.join(tmpdir, os.path.basename(outfile))

		mio.writebmat(c[:,slice(*freqrange)].T, outfile)


def measparser(l):
	'''
	For a line of the form '<index> <filename>', return a tuple of the form
	of (int(<index>), <filename>.strip()).
	'''
	s = l.split(None, 1)
	return (int(s[0]), s[1].strip())


if __name__ == '__main__':
	# Set default options
	lfht, freqrange, tmpdir = 2048, [None], None
	try: nprocs = cpu_count()
	except NotImplementedError: nprocs = 1

	optlist, args = getopt.getopt(sys.argv[1:], 'hl:p:f:t:')

	for opt in optlist:
		if opt[0] == '-l':
			lfht = int(opt[1])
		elif opt[0] == '-p':
			nprocs = int(opt[1])
		elif opt[0] == '-f':
			freqrange = [(int(s, base=10) if len(s) else None) for s in opt[1].split(':')]
		elif opt[0] == '-t':
			tmpdir = opt[1]
		else:
			usage()
			sys.exit(1)

	if len(args) < 1:
		usage()
		sys.exit()

	# Read the file, which specifies a receive channel index and a data file path
	with open(args[0], 'rb') as f:
		measurements = [measparser(l) for l in f.readlines()]

	# Pull out the file names
	infiles = [m[1] for m in measurements]

	try:
		# Spawn processes to handle the inputs
		procs = []
		for i in range(nprocs):
			# Set the function arguments
			args = (infiles[i::nprocs], lfht, freqrange)
			p = Process(target = fhfft, args=args)
			p.start()
			procs.append(p)
		for p in procs: p.join()
	except:
		for p in procs:
			p.terminate()
			p.join()
		raise
