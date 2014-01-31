#!/usr/bin/env python

import numpy as np, os, sys, re, operator as op
from mpi4py import MPI

# Create a record type for the spectral representations
specreptype = np.dtype([('val', np.complex64), ('idx', np.int64)])


def splitspecreps(a):
	'''
	Break a record array a of concatenated spectral representations, with
	dtype [('val', np.complex64), ('idx', np.int64)], into a list of record
	arrays corresponding to each group of spectral representations in the
	original array. The number of records in the first group (output[0]) is
	specified by n[0] = (a[0]['idx'] + 1), with output[0] = a[:n[0]].
		
	The number of records in a subsequent group (output[i]) is given by

		n[i] = (a[sum(n[:i-1])]['idx'] + 1),
		
	with output[i] = a[sum(n[:i-1]):sum(n[:i])].
	'''
	start = 0
	output = []
	while start < len(a):
		nvals = a[start]['idx'] + 1
		if nvals < 1: raise ValueError('Spectral representation counts must be positive')
		grp = a[start:start+nvals]
		if len(grp) < nvals: raise ValueError('Could not read specified number of records')
		output.append(a[start:start+nvals])
		start += nvals
	return output


def countspecreps(f):
	'''
	For a file f, return the number of components for each group of
	spectral representations in the file f. The returns is the list n
	described in the docstring for splitspecreps().
	'''
	# Open the file and determine its size
	infile = open(f, 'rb')
	infile.seek(0, os.SEEK_END)
	fend = infile.tell()
	infile.seek(0, os.SEEK_SET)
	# Scan through the file to pick up all of the counts
	n = []
	while (infile.tell() < fend):
		# Read the header record and add it to the list
		nrec = np.fromfile(infile, dtype=specreptype, count=1)[0]['idx']
		n.append(nrec + 1)
		# Skip over the records for this group
		infile.seek(nrec * specreptype.itemsize, os.SEEK_CUR)

	return n


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


if len(sys.argv) < 4:
	sys.exit('USAGE: %s <prefix> <srcdir> <destdir>' % sys.argv[0])

# Grab the prefix and the source and destination directories
prefix = sys.argv[1]
srcdir = sys.argv[2]
destdir = sys.argv[3]

mpirank, mpisize = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

print 'MPI rank %d of %d: transfer from %s to %s' % (mpirank, mpisize, srcdir, destdir)

# Grab a list of all spectral representations
specfiles = findspecreps(srcdir, prefix=prefix)
grpcounts = [sum(gc) for gc in zip(*[countspecreps(f[0]) for f in specfiles])]
ngroups = len(grpcounts)

print 'MPI rank %d of %d: finished local group counting' % (mpirank, mpisize)

# Create an array to store all of the representations, ordered by group
specreps = np.empty((sum(grpcounts), ), dtype=specreptype)

# Keep track of the offsets for each group as the array is populated
offsets = [0]
for gc in grpcounts[:-1]: offsets.append(offsets[-1] + gc)

for specfile, specidx in specfiles:
	# Read the representations for each source and split by group
	lsreps = splitspecreps(np.fromfile(specfile, dtype=specreptype))
	# Ensure that the number of groups is consistent
	if len(lsreps) != ngroups:
		raise ValueError('Spectral representation group count must not change')
	# Copy each group of samples to the proper place, updating the offsets
	for i, rep in enumerate(lsreps):
		start = offsets[i]
		end = start + len(rep)
		specreps[start:end] = rep
		offsets[i] = end

print 'MPI rank %d of %d: finished local group parsing' % (mpirank, mpisize)

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
rspecreps = np.empty((sum(rcounts), ), dtype=specreptype)
MPI.COMM_WORLD.Alltoallv([specreps, counts, displs, rectype],
		[rspecreps, rcounts, rdispls, rectype])

# Free the MPI type
rectype.Free()

print 'MPI rank %d of %d: finished data exchange' % (mpirank, mpisize)

# Repurpose the specreps to split the received representations
specreps = splitspecreps(rspecreps)

print 'MPI rank %d of %d: finished splitting representations' % (mpirank, mpisize)

# Group the representations according to spectral group
start, share = grpshares[mpirank]
groupreps = [[] for gidx in range(share)]
# Received representations are grouped first by rank, then group, then source
for srclist in srclists:
	for gidx in range(share):
		for src in srclist:
			groupreps[gidx].append((src, specreps.pop(0)))

print 'MPI rank %d of %d: finished grouping representations' % (mpirank, mpisize)

# Ensure that all groups have been assigned
if len(specreps) > 0:
	raise IndexError('Additional received representations were not characterized')

# Write every grouped representation
for grp, reps in zip(range(start, start + share), groupreps):
	# Open an output file
	outpath = os.path.join(destdir, '%sSpecRepsSubvol%d.dat' % (prefix, grp))
	outfile = open(outpath, 'wb')
	# Sort representations by source index before writing
	for rep in sorted(reps, key=op.itemgetter(0)): rep[1].tofile(outfile)
	outfile.close()

print 'MPI rank %d of %d: finished file writes' % (mpirank, mpisize)
