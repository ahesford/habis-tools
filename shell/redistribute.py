#!/usr/bin/env python

import numpy as np, os, sys
import getopt
import socket
import random
import shutil
import subprocess

from itertools import izip, repeat
from mpi4py import MPI

from habis import formats

def usage(progname = 'redistribute.py'):
	binfile = os.path.basename(progname)
	print >> sys.stderr, "Usage:", binfile, "[-h] [-s svolfile] [-n nodelist] [-i host-suffix] <srcdir>/<inprefix> <destdir>/<outprefix>"

if __name__ == '__main__':
	optlist, args = getopt.getopt(sys.argv[1:], 'hi:n:s:')

	svolfile = None
	nodelist = None
	ifsuffix = None

	# Parse the option list
	for opt in optlist:
		if opt[0] == '-n':
			nodelist = [int(l) for l in opt[1].split(',')]
		elif opt[0] == '-s':
			svolfile = opt[1]
		elif opt[0] == '-i':
			ifsuffix = opt[1]
		elif opt[0] == '-h':
			usage(sys.argv[0])
			MPI.COMM_WORLD.Abort()
			sys.exit()
		else:
			usage(sys.argv[0])
			MPI.COMM_WORLD.Abort(1)
			sys.exit('Invalid argument')

	# There must be at least two arguments (input and output file templates)
	if len(args) < 2:
		usage(sys.argv[0])
		MPI.COMM_WORLD.Abort(1)
		sys.exit('Improper argument specification')
	
	# Grab the in prefix and the source directory
	srcdir, inprefix = os.path.split(args[0])
	# The destination directory and prefix can remain joined together
	destform = args[1]

	mpirank, mpisize = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
	identifier = 'MPI rank %d of %d' % (mpirank, mpisize)

	print '%s: transfer from %s to %s' % (identifier, srcdir, os.path.dirname(destform))

	# Grab a dictionary of all spectral representations, keyed by index
	hostname = socket.gethostname()
	if ifsuffix: hostname += ifsuffix
	specfiles = dict((s[1], hostname + ":" + s[0]) 
			for s in formats.findenumfiles(srcdir, prefix=inprefix, suffix='\.dat'))

	# Accumulate the list of sources on every rank, and flatten into one dictionary
	srclists = MPI.COMM_WORLD.allgather(specfiles)
	srclists = dict(kv for s in srclists for kv in s.items())

	# If a subvolume list was specified, use it; otheruse use all subvolumes
	if svolfile: svols = np.loadtxt(svolfile).astype(int).tolist()
	else: svols = sorted(srclists.keys())

	# If a receiving nodelist was not specified, use all participating nodes
	if nodelist is None: nodelist = range(mpisize)

	# Figure the share of subvolumes to be received at this rank
	share, rem = len(svols) / len(nodelist), len(svols) % len(nodelist)
	try:
		drank = nodelist.index(mpirank)
		start = drank * share + min(drank, rem)
		if drank < rem: share += 1
		rcvsvols = svols[start:start+share]
	except ValueError:
		rcvsvols = []

	# Randomly shuffle the subvolume ordering to minimize contention
	random.shuffle(rcvsvols)

	# Pull out a list of local subvolumes
	locsvols = set(specfiles.keys())

	# Transfer the subvolumes
	for svol in rcvsvols:
		dstfile = destform + str(svol) + '.dat'
		if svol in locsvols:
			# If the file is local, just copy it
			srcfile = os.path.join(srcdir, os.path.basename(specfiles[svol]))
			shutil.copyfile(srcfile, dstfile)
		else:
			# Otherwise, use scp
			srcfile = srclists[svol]
			subprocess.call(["scp", srcfile, dstfile])

	print '%s: finished data exchange' % identifier
