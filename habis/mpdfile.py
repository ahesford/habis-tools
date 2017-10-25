'''
Routines for accessing local data in a distributed, MPI environment
'''

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import hashlib


def getatimes(atfile, elements, column=0, backscatter=True,
		vclip=None, mask_outliers=False, start=0, stride=1):
	'''
	Read the 2-key arrival-time map with name atfile and filter the map to
	include only keys such that each index is a key in the mapping elements
	from indices to element locations, and the average speed (the propagation
	path length, computed from element locations, divided by the arrival
	time) falls between vclip[0] and vclip[1].

	The column argument specifies which index in multi-value arrival-time
	maps should be selected.

	If backscatter is False, backscatter arrival times---those values for
	keys of the form (t, t)---are removed from the arrival-time map.
	Otherwise, backscatter measurements will remain in the map if they are
	encoded in the file. Backscatter measurements are not filtered by vclip
	because an average speed cannot be computed for backscatter
	configurations.

	If mask_outliers is specified, it should either be a Boolean True or a
	positive floating-point value. Arrival times will be excluded if they
	correspond to average speeds that fall more than M * IQR below the
	first quartile or more than M * IQR above the third quartile, where M
	is 1.5 if mask_outliers is a Boolean True and M is the floating-point
	value of mask_outliers otherwise. Outlier exclusion does not apply to
	backscatter times.

	Only every stride-th *valid* record, starting with the start-th record,
	is retained.
	'''
	from .formats import loadkeymat
	from numpy.linalg import norm
	from numpy import percentile

	if not 0 <= start < stride:
		raise ValueError('Index start must be at least zero and less than stride')

	if vclip:
		if len(vclip) != 2:
			raise ValueError('Argument "vclip" must specify two values')
		vclip = sorted(vclip)

	# Load the map, eliminate invalid elemenets, and keep the right portion
	atimes = { }
	idx = 0

	if mask_outliers:
		# Record the element distances and all speed values for filtering
		eldists = { }
		spdvals = [ ]

	for (t, r), v in loadkeymat(atfile, nkeys=2, scalar=False).items():
		# Skip backscatter if possible
		if not backscatter and t == r: continue

		try: elt, elr = elements[t], elements[r]
		except KeyError: continue

		time = v[column]

		if t != r:
			# Compute average speed
			trdist = norm(elt - elr)
			aspd = trdist / time
			if vclip and not vclip[0] <= aspd <= vclip[1]: continue
			if mask_outliers:
				eldists[t,r] = trdist
				spdvals.append(aspd)

		# Keep every stride-th valid record
		if idx % stride == start: atimes[t,r] = time
		# Increment the valid record count
		idx += 1

	if not mask_outliers: return atimes

	if isinstance(mask_outliers, bool): mask_outliers = 1.5
	else: mask_outliers = float(mask_outliers)

	# Define outlier limits
	q1, q3 = percentile(spdvals, [25, 75])
	iqr = q3 - q1
	lo, hi = q1 - mask_outliers * iqr, q3 + mask_outliers * iqr

	# Filter arrival times to remove outliers
	return { (t, r): v for (t, r), v in atimes.items()
			if t == r or lo <= eldists[t,r] / v <= hi }



def sha512(fname, msize=10):
	'''
	For the named file (which should not be open), compute the SHA-512 sum,
	reading in chunks of msize megabytes. The return value is the output of
	hexdigest() for the SHA-512 Hash object.
	'''
	bsize = int(msize * 2**20)
	if bsize < 1: raise ValueError('Specified chunk size too small')

	cs = hashlib.sha512()
	with open(fname, 'rb') as f:
		while True:
			block = f.read(bsize)
			if not block: break
			cs.update(block)

	return cs.hexdigest()

def fhashmap(fnames, msize=10):
	'''
	For each file in the given list of files, each of which should provide
	the name of an unopened but readable file, compute the SHA-512 sum by
	opening and reading the file in chunks of msize megabytes.

	The return value is a map of the form (shasum -> file name).
	'''
	return { sha512(f): f for f in fnames }


def frankmap(fhmap, comm):
	'''
	Given fhmap, a rank-local map of the form (shasum -> file name) as
	produced by fhashmap, gather the maps on all ranks in the mpi4py
	communicator comm (using comm.allgather) and produce a composite map of
	the form (shasum -> rank list), where rank list is a list of all ranks
	in comm that have local access to the file with SHA-512 sum shasum.

	Each list of ranks that can access a given file will be sorted in
	increasing rank order.
	'''
	rankmap = { }
	for rank, csums in enumerate(comm.allgather(list(fhmap.keys()))):
		for cs in csums:
			try: rankmap[cs].append(rank)
			except KeyError: rankmap[cs] = [rank]
	return rankmap


def flocshare(fnames, comm, msize=10):
	'''
	Given a list fnames of names for unopened, locally readable files,
	produce a file hash map fhmap = fhashmap(fnames, msize) and a rank map
	frmap = frankmap(fhmap, comm), and return a new map of the form

		[fname -> (index, length)],

	where index satisfies frmap[s][index] == comm.rank for s such that
	fhmap[s] == fname, and length is the length of frmap[s].
	'''
	fhmap = fhashmap(fnames, msize)
	frmap = frankmap(fhmap, comm)
	rank = comm.rank

	lmap = { }
	for cs, fn in fhmap.items():
		# Find the index of this rank in the rank list
		try: rlist = frmap[cs]
		except KeyError:
			raise ValueError('File %s exists locally, but not in rank map' % (fn,))

		try: index = rlist.index(rank)
		except ValueError:
			raise ValueError('File %s exists locally, but list of ranks does nto contain it' % (fn,))

		lmap[fn] = (index, len(rlist))

	return lmap
