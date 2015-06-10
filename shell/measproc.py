#!/usr/bin/env python

import numpy as np, os, sys, fht, pyfftw, getopt, operator as op
from collections import defaultdict
from mpi4py import MPI

from pycwp import mio, process


def usage(progname=None):
	if progname is None: progname = sys.argv[0]
	binfile = os.path.basename(progname)
	print >> sys.stderr, 'USAGE: %s [-l l] [-p p] [-f start:end:skip] [-t tmpdir] <measurements> <outfmt>' % binfile
	print >> sys.stderr, '''
  Preprocess HABIS measurement data by Hadamard decoding transmissions and
  Fourier transforming the time-domain data. Measurement data is listed in the
  'measurements' file which specifies, on each line, the index of a receiver
  channel followed by a path to the file name containing a full set of
  time-domain data for the channel.

  The processed data is transposed between nodes so that, on output, each node
  contains all channel measurements for a subset of the transmissions included
  in each measurement input. The output data is saved to files with a name
  format given in outfmt, which is used to generate output files named
  outfmt.format(tidx) for each integer transmission index tidx.

  OPTIONAL ARGUMENTS:
  -p: Use p processors (default: all available processors)
  -l: Assume Hadamard coding in groups of l elements (default: 2048)
  -f: Retain only FFT frequency bins in slice(start, end, skip) (default: all)
  -t: Store spectral output files in tmpdir (default: same as measurements)
	'''


def fhfft(infiles, lfht, freqrange=[None], tmpdir=None):
	'''
	For real, N x M matrices A stored in the binary files infiles, tile a
	series of Hadamard transforms of length lfht along the M dimension, and
	then perform a DFT along the N dimension to produce an output matrix B.
	The sub-matrix B[S,:], where S = slice(*freqrange) is saved in a file
	whose name is determined by appending '.spectral' to the corresponding
	value in infiles.

	The length of the Hadamard transforms must be a power of 2, and M must
	be an integer multiple of the Hadamard transform length.
	'''
	if not fht.ispow2(lfht):
		raise ValueError('Hadamard transform length must be a power of 2')

	nfft, nfht = 0, 0
	
	for infile in infiles:
		# Read the input file
		a = mio.readbmat(infile)
		if a.dtype != np.float32 and not a.dtype != np.float64:
			raise TypeError('Matrix must have a dtype of float32 or float64')

		if a.shape != (nfft, nfht):
			nfft, nfht = a.shape
			if nfht % lfht != 0:
				raise ValueError('Second matrix dimension must be integer multiple of Hadamard transform length')
		
			# Create an intermediate array to store Hadamard results
			# The FFT axis should be contiguous here FFT performance
			b = pyfftw.n_byte_align_empty((nfft, nfht), pyfftw.simd_alignment, a.dtype, order='F')
		
			# Create an output array to store the final FFT results
			# The length of the FFT axis is half the input, plus one for DC
			cdim = (nfft // 2 + 1, nfht)
			cdtype = np.complex64 if (a.dtype == np.float32) else np.complex128
			c = pyfftw.n_byte_align_empty(cdim, pyfftw.simd_alignment, cdtype, order='F')
		
			# Create an FFT plan before populating the Hadamard results
			fwdfft = pyfftw.FFTW(b, c, axes=(0,))
		
		# Reorder the input so that the FHT axis is continguous
		a = np.reshape(a.flat, a.shape)

		# At this point, the gain compensation should be applied

		# Perform the tiled Hadamard transforms
		for s in range(0, nfht, lfht):
			e = s + lfht
			b[:,s:e] = fht.fht(a[:,s:e], axes=1)

		# At this point, time-gating should be applied

		# Perform the FFT
		fwdfft()

		# Write the spectral output
		outfile = specfilename(infile, tmpdir)
		mio.writebmat(c[slice(*freqrange),:], outfile)


def specfilename(infile, tmpdir = None):
	'''
	Given an input file at a path infile and an optional temporary
	directory, return an output file whose name is the input file name with
	'.spectral' appended. If tmpdir is not None, replace the path to the
	input file with the path to tmpdir.
	'''
	outfile = infile + '.spectral'
	if tmpdir: outfile = os.path.join(tmpdir, os.path.basename(outfile))
	return outfile


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
	nprocs = process.preferred_process_count()

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

	if len(args) < 2:
		usage()
		sys.exit()

	# Read the input file for receive channel indices and file paths
	with open(args[0], 'rb') as f:
		measurements = [measparser(l) for l in f.readlines()]

	# Store the desired output format
	outfmt = args[1]

	# Pull out the file names and process each file in parallel
	with process.ProcessPool() as pool:
		infiles = [m[1] for m in measurements]
		for i in range(nprocs):
			args = (infiles[i::nprocs], lfht)
			kwargs = {'freqrange': freqrange, 'tmpdir': tmpdir}
			pool.addtask(target = fhfft, args=args, kwargs=kwargs)
		pool.start()
		pool.wait()

	# Now read each processed file in sequence and store in the overall matrix
	for i, (ridx, recvfile) in enumerate(measurements):
		recvdata = mio.readbmat(specfilename(recvfile, tmpdir))
		# Store the data in its specific slot in the overall matrix
		try: measmat[:,i,:] = recvdata.T
		except NameError:
			# If the matrix has not been created, create and then store
			# FFT samples most rapidly varying, source varies least rapidly
			nfft, nsrc = recvdata.shape
			nmeas = len(measurements)
			dtype = recvdata.dtype
			measmat = np.empty((nsrc, nmeas, nfft), dtype=dtype, order='C')
			measmat[:,i,:] = recvdata.T

	# Ensure the source and FFT sample counts are compatible across proceses
	nsrc, nmeas, nfft = measmat.shape
	mshapes = MPI.COMM_WORLD.allgather((nsrc, nfft))
	for shape in mshapes[1:]:
		if shape != mshapes[0]:
			raise ValueError('All processes must have identical source and FFT sample counts')

	# Figure the share of sources to be received at each rank
	mpirank, mpisize = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
	share, rem = nsrc / mpisize, nsrc % mpisize
	# Each entry takes the form (start, share)
	srcshares = [(i * share + min(i, rem), share + int(i < rem)) for i in range(mpisize)]

	# Compute the displacements and counts to be sent to each node
	nsrcrec = nmeas * nfft
	counts, displs = zip(*[(l * nsrcrec, s * nsrcrec) for s, l in srcshares])

	# Accumulate the list of counts coming from every rank
	rcounts = MPI.COMM_WORLD.alltoall(counts)
	# Build the corresponding receive displacements
	rdispls = [0]
	for rc in rcounts[:-1]: rdispls.append(rdispls[-1] + rc)

	# Allocate an array to store received data and exchange it
	rmeasmat = np.empty((sum(rcounts),), dtype=measmat.dtype)

	if measmat.dtype == np.complex64: mpidtype = MPI.COMPLEX
	elif measmat.dtype == np.complex128: mpidtype = MPI.DOUBLE_COMPLEX
	else: raise TypeError('Exchanged data must be single- or double-precision complex')

	MPI.COMM_WORLD.Alltoallv([measmat.ravel('C'), counts, displs, mpidtype],
			[rmeasmat, rcounts, rdispls, mpidtype])

	# Gather the list of receiver channels from each rank for sorting
	recvlists = MPI.COMM_WORLD.allgather([m[0] for m in measurements])
	nrecv = sum(len(r) for r in recvlists)

	start, share = srcshares[mpirank]
	# Reshape so rows are source and reciever folded and colums are FFT indices
	rmeasmat = rmeasmat.reshape((nrecv * share, nfft), order='C')

	# Build mappings for source channel, receiver channel and row index
	indices, ridx = defaultdict(list), 0
	for recvs in recvlists:
		for tidx in range(start, start + share):
			for recv in recvs:
				indices[tidx].append((recv, ridx))
				ridx += 1

	# Save each output file in turn
	for tidx, recvs in indices.iteritems():
		# Sort the measurement row indices according to receiver channel
		rdata = rmeasmat[[r[1] for r in sorted(recvs)], :]
		mio.writebmat(rdata, outfmt.format(tidx))
