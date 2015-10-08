#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, os, sys, re

from multiprocessing import Pool

from habis import formats

def usage(progname):
	print >> sys.stderr, 'USAGE (forward): %s <prefix> <srcdir> <destdir> ... <destdir>' % progname
	print >> sys.stderr, '      (reverse): %s -r <prefix> <srcdir> ... <srcdir> <destdir>' % progname
	print >> sys.stderr, ''
	print >> sys.stderr, '  To force forward mode when prefix is "-r", specify "--" before <prefix>'

	sys.exit(1)


def batchspecrep(rep, nbatch = 1):
	'''
	Break the record array rep of concatenated spectral representations,
	with dtype [('val', np.complex64), ('idx', np.int64)], into nbatch
	subarrays of approximately equal size, returning the subarrays in a
	list.
	
	The first record rep[0]['idx'] == (len(rep) - 1); each output subarray
	has a prepended header record such that
	
		output[i][0]['idx'] == len(output[i]) - 1

	and output[i][0]['val'] == rep[0]['val'].
	'''
	if nbatch < 1: raise ValueError('Number of batches must be at least one')
	# Handle the trivial case in which nbatch is one
	if nbatch == 1: return [rep]

	# Figure the number of records in each block
	numrecs = len(rep) - 1
	blocksizes = [numrecs // nbatch + int(i < numrecs % nbatch) for i in range(nbatch)]
	dtype = formats.specreptype()
	# Build the output arrays; skip header in the input
	output, start = [], 1
	for i, bs in enumerate(blocksizes):
		# Create the output for this block, include a new header
		output.append(np.empty((bs + 1,), dtype=dtype))
		# Copy the new header, replacing the index value
		output[-1][0] = rep[0]
		output[-1][0]['idx'] = bs
		# Copy the data
		end = start + bs
		output[-1][1:] = rep[start:end]
		start = end

	return output


def mergespecrep(reps):
	'''
	Given a collection of spectral representations, return a record array
	which is the concatenation of all samples in all representations.
	Because the first record of each representation is a header specifying
	the representation length, eliminate the headers in the concatenation
	and specify a new header in the output.
	'''
	# Compute the total number of samples in the merged specrep
	replen = sum(rep[0]['idx'] for rep in reps)
	# Make the output array with room for a header
	output = np.empty((replen + 1,), dtype=formats.specreptype())
	# Concatenate the inputs, skipping each header
	output[1:] = np.concatenate([rep[1:] for rep in reps])
	# Create the header in the output, overwriting the size
	output[0] = reps[0][0]
	output[0]['idx'] = replen

	return output


def batchrepfile(args):
	'''
	Split each of the concatenated record arrays listed in specfile in the
	directory srcdir into approximately equal sized pieces (one for each
	output directory in destdirs). Then concatenate corresponding pieces
	from each record array and store the result to a file with the same
	name in the corresponding destination directory.
	'''
	specfile, srcdir, destdirs = args
	nbatch = len(destdirs)
	if nbatch < 1:
		raise ValueError('At least one output directory must be specified')

	# Get the name of the input file
	filename = os.path.join(srcdir, specfile)

	# Read the spectral representations from the input
	lsreps = np.fromfile(filename, dtype=formats.specreptype())
	# Break the representations by group and then batch each group
	batchreps = [batchspecrep(r, nbatch) for r in formats.splitspecreps(lsreps)]
	# Now concatenate representations for each batch and write to the output
	for br, dr in zip(zip(*batchreps), destdirs):
		np.concatenate(br).tofile(os.path.join(dr, specfile))
	print 'Split file', specfile


def mergerepfile(args):
	'''
	Read the concatenated record arrays listed in a file named specfile in
	each of the directories srcdirs and merge corresponding arrays. Store
	the concatenation of the merged arrays in a file named specfile in the
	directory destdir.
	'''
	specfile, srcdirs, destdir = args
	# Read all batched spectral representations and split the representations
	dtype = formats.specreptype()
	lsreps = [formats.splitspecreps(
			np.fromfile(os.path.join(srcdir, specfile), dtype=dtype))
			for srcdir in srcdirs]
	# Merge corresponding groups and write the output
	mergereps = [mergespecrep(reps) for reps in zip(*lsreps)]
	np.concatenate(mergereps).tofile(os.path.join(destdir, specfile))
	print 'Merged file', specfile


def findspecreps(dir, prefix='.*SpecRepsElem'):
	'''
	Find all files in the directory dir with a name matching the regexp
	r'^<PREFIX>SpecRepsElem([0-9]+).dat$', where <PREFIX> is replaced with
	an optional prefix to restrict the search, and return a list of tuples
	in which the first item is the name and the second item is the matched
	integer.
	'''

	# Build the regexp and filter the list of files in the directory
	regexp = re.compile(r'^%s([0-9]+).dat$' % prefix)
	return [(f, int(m.group(1)))
		for f in os.listdir(dir)
		for m in [regexp.match(f)] if m]


if __name__ == '__main__':
	# Perform preliminary argument checking
	if len(sys.argv) < 4: usage(sys.argv[0])

	if sys.argv[1] == '-r' or sys.argv[1] == '--':
		# Force reverse mode if '-r' is specified
		fwdmode = not (sys.argv[1] == '-r')
		# Ensure enough arguments
		if len(sys.argv) < 5: usage(sys.argv[0])
		# Grab the file prefix and the source and destination directories
		prefix = sys.argv[2]
		srcdirs = sys.argv[3:-1]
		destdir = sys.argv[-1]
	else:
		fwdmode = True
		# Ensure enough arguments
		if len(sys.argv) < 4: usage(sys.argv[0])
		# Grab the file prefix and the source and destination directories
		prefix = sys.argv[1]
		srcdir = sys.argv[2]
		destdirs = sys.argv[3:]

	# Batching only makes sense if there are 2 or more chunks
	if (fwdmode and len(destdirs) < 2) or (not fwdmode and len(srcdirs) < 2):
		sys.exit('Batching is just a copy when nbatch is one')

	# Find the files for batch processing
	# In reverse mode, file names are pulled from the first source directory
	files = []
	if fwdmode: files = findspecreps(srcdir, prefix)
	else: files = findspecreps(srcdirs[0], prefix)

	if len(files) < 1: sys.exit('No files found for batching')

	# Create a multiprocessing pool
	p = Pool()

	if fwdmode: p.map(batchrepfile, ((f[0], srcdir, destdirs) for f in files))
	else: p.map(mergerepfile, ((f[0], srcdirs, destdir) for f in files))
