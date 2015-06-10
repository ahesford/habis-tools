#!/usr/bin/env python

import sys, os
import getopt
import numpy as np

from subprocess import call
from mpi4py import MPI
from pycwp import util, process

def usage(progname = 'gpuexec.py'):
	binfile = os.path.basename(progname)
	print >> sys.stderr, "Usage:", binfile, "[-h] [-m] [-r] <-b blocks | -l blockfile> <gpus> <command> [args] ..."

def procexec(args):
	'''
	For a numerical index idx, execute in a subprocess
	
		args[0] index *args[1:].
	'''
	# There is no work to undertake if there is no executable
	if len(args) < 1: return
	# Call the program wth the inputs converted to strings
	call([str(a) for a in args])

if __name__ == '__main__':
	# Set default options
	blocklist, blockfile = None, None
	multiblock, userank = False, False

	# Parse the command-line arguments
	optlist, args = getopt.getopt(sys.argv[1:], 'hmrb:l:')
	for opt in optlist:
		if opt[0] == '-m':
			multiblock = True
		elif opt[0] == '-r':
			userank = True
		elif opt[0] == '-b':
			blocklist = range(int(opt[1]))
		elif opt[0] == '-l':
			blockfile = opt[1]
		elif opt[0] == '-h':
			usage(sys.argv[0])
			MPI.COMM_WORLD.Abort()
			sys.exit()
		else:
			usage(sys.argv[0])
			MPI.COMM_WORLD.Abort(1)
			sys.exit('Invalid argument')

	if len(args) < 2 or ((blocklist is None) == (blockfile is None)):
		usage(sys.argv[0])
		MPI.COMM_WORLD.Abort(1)
		sys.exit('Improper argument specification')

	# Determine the process rank and the size
	rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

	# Grab the list of GPUs to use
	gpus = [int(s.strip()) for s in args[0].split(',')]
	ngpus = len(gpus)

	# If a block file was specified, read the blocklist
	if blockfile: blocklist = np.loadtxt(blockfile).astype(int).tolist()

	# Grab the remaining arguments
	args = args[1:]

	# Build the blocklist and grab the local share
	share = len(blocklist) / size
	rem = len(blocklist) % size
	start = rank * share + min(rank, rem)
	if rank < rem: share += 1
	localblocks = blocklist[start:start+share]

	# If the "multiblock" option is specified, the block argument consists
	# of multiple comma-separated block indices; invoke exactly one process
	# per GPU, each with a multi-block argument
	if multiblock:
		localblocks = [','.join(str(b) for b in localblocks[i::ngpus]) for i in range(ngpus)]

	for blocks in util.grouplist(localblocks, ngpus):
		with process.ProcessPool() as pool:
			for b, g in zip(blocks, gpus):
				# The blocking arguments may include a node rank
				bargs = [g, b] + ([rank] if userank else [])
				pargs = args[:1] + bargs + args[1:]
				pool.addtask(target=procexec, args=(pargs,))
			pool.start()
			pool.wait()
