#!/usr/bin/env python

import numpy as np, os, sys, re, operator as op
from itertools import izip, repeat
from mpi4py import MPI

from habis import formats

def findspecreps(dir, prefix='.*'):
	'''
	Find all files in the directory dir with a name matching the regexp
	r'^<PREFIX>SpecRepsElem([0-9]+).dat$', where <PREFIX> is replaced with
	an optional prefix to restrict the search, and return a list of tuples
	in which the first item is the name and the second item is the matched
	integer.
	'''

	# Build the regexp and filter the list of files in the directory
	regexp = re.compile(r'^%sSpecRepsElem([0-9]+).dat$' % prefix)
	return [(os.path.join(dir, f), int(m.group(1)))
			for f in os.listdir(dir) for m in [regexp.match(f)] if m]

if __name__ == '__main__':
	if len(sys.argv) < 4:
		sys.exit('USAGE: %s <prefix> <srcdir> <destdir>' % sys.argv[0])
	
	# Grab the prefix and the source and destination directories
	prefix = sys.argv[1]
	srcdir = sys.argv[2]
	destdir = sys.argv[3]

	mpirank, mpisize = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
	identifier = 'MPI rank %d of %d' % (mpirank, mpisize)

	print '%s: transfer from %s to %s' % (identifier, srcdir, destdir)

	# Grab a list of all spectral representations
	specfiles = findspecreps(srcdir, prefix=prefix)
	grpcounts = [sum(gc) for gc in zip(*[formats.countspecreps(f[0]) for f in specfiles])]
	ngroups = len(grpcounts)

	print '%s: finished local group counting' % identifier

	dtype = formats.specreptype()

	# Create an array to store all of the representations, ordered by group
	specreps = np.empty((sum(grpcounts), ), dtype=dtype)

	# Keep track of the offsets for each group as the array is populated
	offsets = [0]
	for gc in grpcounts[:-1]: offsets.append(offsets[-1] + gc)

	for specfile, specidx in specfiles:
		# Read the representations for each source and split by group
		lsreps = formats.splitspecreps(np.fromfile(specfile, dtype=dtype))
		# Ensure that the number of groups is consistent
		if len(lsreps) != ngroups:
			raise ValueError('Spectral representation group count must not change')
		# Copy each group of samples to the proper place, updating the offsets
		for i, rep in enumerate(lsreps):
			start = offsets[i]
			end = start + len(rep)
			specreps[start:end] = rep
			offsets[i] = end

	print '%s: finished local group parsing' % identifier

	# Determine the group shares for each MPI rank
	share, rem = ngroups / mpisize, ngroups % mpisize
	# Each entry takes the form (start, share)
	grpshares = [(i * share + min(i, rem), share + int(i < rem)) for i in range(mpisize)]

	# Compute the displacements and counts to be sent to each node
	counts = []
	displs = [0]
	for s, l in grpshares:
		counts.append(sum(grpcounts[s:s+l]))
		displs.append(displs[-1] + counts[-1])
	# There is an extra displacement that should now be striped
	displs[-1:] = []

	# Accumulate the list of sources on every rank
	srclists = MPI.COMM_WORLD.allgather([s[1] for s in specfiles])

	# Accumulate the list of counts coming from every rank
	rcounts = MPI.COMM_WORLD.alltoall(counts)
	# Build the receive displacements
	rdispls = [0]
	for rc in rcounts[:-1]: rdispls.append(rdispls[-1] + rc)

	# Create an MPI datatype corresponding to the spectral representation record
	rectype = MPI.Datatype.Create_struct((1,1),
			(0, MPI.COMPLEX.size), (MPI.COMPLEX, MPI.LONG))
	rectype.Commit()

	# Allocate an array to store received representations and exchange them
	rspecreps = np.empty((sum(rcounts), ), dtype=dtype)
	MPI.COMM_WORLD.Alltoallv([specreps, counts, displs, rectype],
			[rspecreps, rcounts, rdispls, rectype])

	# Free the MPI type
	rectype.Free()

	print '%s: finished data exchange' % identifier

	# Figure the local portion of groups that were received
	start, share = grpshares[mpirank]

	# Enumerate the group and source indices for sorting of received representations
	indices = [(gidx, src) for srcs in srclists for gidx in range(share) for src in srcs]

	# Sort by group, then by source index
	specreps = sorted(izip(indices, formats.splitspecreps(rspecreps)), key=op.itemgetter(0))

	print '%s: finished splitting and sorting representations' % identifier

	fname = lambda grp: os.path.join(destdir, '%sSpecRepsSubvol%d.dat' % (prefix, grp))

	# Open files for every output group
	groupfiles = [open(fname(grp + start), 'wb') for grp in range(share)]

	# Write every grouped representation
	for (gidx, src), rep in specreps:
		# Save the representation in the file for the indicated group
		rep.tofile(groupfiles[gidx])

	for gf in groupfiles: gf.close()

	print '%s: finished file writes' % identifier
