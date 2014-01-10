#!/opt/epd-7.3-2-rh5-x86_64/bin/python

import sys, os
from subprocess import call
from multiprocessing import Process
from mpi4py import MPI
from pyajh import util

def procexec(args):
	'''
	For a numerical index idx, execute in a subprocess

		args[0] index *args[1:].
	'''
	# There is no work to undertake if there is no executable
	if len(args) < 1: return
	# Call the program wth the inputs converted to strings
	call([str(a) for a in args])


if len(sys.argv) < 3:
	sys.exit('USAGE: %s <gpus> <command> [args] ...' % sys.argv[0])

# Determine the process rank and the size
rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

# Grab the list of GPUs to use
gpus = [int(s.strip()) for s in sys.argv[1].split(',')]
ngpus = len(gpus)

# Grab the remaining arguments
args = sys.argv[2:]

# Build the blocklist and grab the local share
blocklist = range(80)
share = len(blocklist) / size
rem = len(blocklist) % size
start = rank * share + min(rank, rem)
if rank < rem: share += 1
localblocks = blocklist[start:start+share]

for blocks in util.grouplist(localblocks, ngpus):
	procs = []
	try:
		for b, g in zip(blocks, gpus):
			pargs = args[:1] + [g, b] + args[1:]
			p = Process(target=procexec, args=(pargs,))
			p.start()
			procs.append(p)
		for p in procs: p.join()
	except:
		for p in procs:
			p.terimate()
			p.join()
		raise
