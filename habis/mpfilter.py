'''
Routines for distributed filtering of Numpy arrays in an MPI environment.
'''

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np
import scipy.ndimage

def parshare(n, overlap, rank, size):
	'''
	For a sequence of n values, return as (start, end) the indices that
	define the rank-th of size chunks of the sequence that overlap by
	overlap items.
	'''
	# Find the size and start of non-overlapping portions
	share, srem = n // size, n % size
	start = rank * share + min(rank, srem)
	if rank < srem: share += 1

	# Extend ends to overlap reigion
	return max(0, start - overlap), min(n, start + share + overlap)


def gathersize(n, nrec, size):
	'''
	For a sequence of n * nrec items represented as n contiguous blocks of
	nrec items each, return as (starts, shares) lists of the starting
	indices (into the flat list) and item counts for each of size chunks.
	'''
	share, srem = n // size, n % size
	starts, shares = zip(*((nrec * (share * i + min(i, srem)),
				nrec * (share + int(i < srem))) for i in range(size)))
	return starts, shares


def parfilter(name, comm=None):
	'''
	For a given named filter, construct a MPI distributed version of the
	filter that divides the workload along the first axis of the filtered
	image among all ranks in the MPI communicator comm (MPI.COMM_WORLD by
	default). The filter must support a "size" or "footprint" argument
	because distributed slices of the image to be filtered will overlap by
	half the filter (or footprint) size along the first axis to avoid
	boundary artifacts that would otherwise result from the slicing.

	The filter that will be parallelized is selected from scipy.ndimage if
	such a function exists in that module, or else from pycwp.filter.

	The return value is a function with the same signature as the original
	filter. The function will handle distribution of the input array (which
	should be the entire array on each process) and accumulation of the
	result (which will be the same array on each process).
	'''
	try: filt = getattr(scipy.ndimage, name)
	except AttributeError:
		import pycwp.filter
		filt = getattr(pycwp.filter, name)

	from mpi4py import MPI
	if comm is None: comm = MPI.COMM_WORLD

	def filterfunc(a, size=None, footprint=None, *args, **kwargs):
		# Determine the necessary overlap in slicing
		if footprint is not None:
			footprint = np.asarray(footprint)
			npad = footprint.shape[0] // 2
			kwargs['footprint'] = footprint
		elif size is not None:
			try: npad = size[0] // 2
			except TypeError: npad = size // 2
			kwargs['size'] = size
		else: raise TypeError('One of "size" or "footprint" is required')

		# Make sure the array is an array
		a = np.asarray(a)

		# Pull the local portion with appropriate padding for filtering
		lidx, hidx = parshare(a.shape[0], npad, comm.rank, comm.size)

		# Filter and update the local portion of a
		lfa = filt(a[lidx:hidx], **kwargs)

		# Make sure that the output array is compatible
		b = np.zeros(a.shape[:1] + lfa.shape[1:], dtype=np.float64, order='C')
		b[lidx:hidx] = lfa

		# Gather local contributions everywhere
		nrec = int(np.prod(b.shape[1:]))
		starts, shares = gathersize(b.shape[0], nrec, comm.size)
		comm.Allgatherv(MPI.IN_PLACE, [b, shares, starts, MPI.DOUBLE])

		# Return the entire result
		return b

	return filterfunc
