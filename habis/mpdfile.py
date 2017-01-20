'''
Routines for accessing local data in a distributed, MPI environment
'''

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import hashlib

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
	return { sha512sum(f): f for f in fnames }


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
	for rank, csums in enumerate(comm.allgather(fhmap.keys())):
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
	for cs, fn in fhmap.iteritems():
		# Find the index of this rank in the rank list
		try: rlist = frmap[cs]
		except KeyError:
			raise ValueError('File %s exists locally, but not in rank map' % (fn,))

		try: index = rlist.index(rank)
		except ValueError:
			raise ValueError('File %s exists locally, but list of ranks does nto contain it' % (fn,))

		lmap[fn] = (index, len(rlist))

	return lmap
