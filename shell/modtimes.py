#!/usr/bin/env python

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, os

from mpi4py import MPI

import numpy as np

from pycwp.cytools.interpolator import LinearInterpolator3D
from pycwp.cytools.eikonal import FastSweep

from habis.pathtracer import PathTracer

from habis.formats import loadkeymat as ldkmat, loadmatlist as ldmats, savez_keymat
from habis.habiconf import HabisConfigParser, HabisConfigError, matchfiles


def usage(progname=None, retcode=1):
	if not progname: progname = os.path.basename(sys.argv[0])
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname
	sys.exit(int(retcode))


if __name__ == '__main__':
	if len(sys.argv) != 2: usage()

	try:
		config = HabisConfigParser(sys.argv[1])
	except Exception as e:
		err = 'Unable to load configuration file %s' % sys.argv[1]
		raise HabisConfigError.fromException(err, e)

	rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

	# Try to build a tracer from the configuration
	tracer = PathTracer.fromconf(config)
	bx = tracer.box

	tsec = 'modtimes'
	def _throw(msg, e):
		errmsg = msg + ' in [' + tsec + ']'
		raise HabisConfigError.fromException(errmsg, e)

	try:
		# Find the slowness map
		s = np.load(config.get(tsec, 'slowness')).astype(np.float64)
		if s.shape != bx.ncell:
			raise ValueError('Shape of slowness must be %s' % (bx.ncell,))
	except Exception as e: _throw('Configuration must specify slowness', e)

	# Read the optional default slowness
	try: slowdef = config.get(tsec, 'slowdef', mapper=float, default=None)
	except Exception as e: _throw('Invalid optional slowdef', e)

	try:
		# Load element files
		efiles = matchfiles(config.getlist(tsec, 'elements'))
		elements = ldmats(efiles, nkeys=1)
	except Exception as e: _throw('Configuration must specify elements', e)

	try:
		# Load target files
		tfiles = matchfiles(config.getlist(tsec, 'targets'))
		targets = ldmats(tfiles, nkeys=1)
	except Exception as e: _throw('Configuration must specify targets', e)

	# Read output file location
	try: output = config.get(tsec, 'output')
	except Exception as e: _throw('Configuration must specify output', e)

	# Use Eikonal solutions instead of tracing, if desired
	try: eikonal = config.get(tsec, 'eikonal', mapper=bool, default=False)
	except Exception as e: _throw('Invalid optional eikonal', e)

	trpairs = sorted((t, r) for t in elements for r in targets)

	MPI.COMM_WORLD.Barrier()

	# Determine the local share of transmit-receive pairs
	npairs = len(trpairs)
	share, srem = npairs / size, npairs % size
	start = rank * share + min(rank, srem)
	if rank < srem: share += 1

	# Build an interpolator for the slowness map
	si = LinearInterpolator3D(s)
	si.default = slowdef

	if eikonal:
		# Prepare the Eikonal solver
		eik = FastSweep(bx)
		# Track the solution for the last transmission
		lt, tmi = None, None
		if not rank: print 'Using Eikonal solution for arrival times'

	# Calculate local share of transmit-receive times
	atimes = { }
	ipow = 1

	for i, (t, r) in enumerate(trpairs[start:start+share]):
		src, rcv = elements[t], targets[r]

		if not eikonal:
			# Use path tracing; only the path integral matters
			atimes[t,r] = tracer.trace(si, src, rcv, intonly=True)
		else:
			# Use the Eikonal solution
			if t != lt or tmi is None:
				# Compute interpolated solution for a new transmitter
				tmi = LinearInterpolator3D(eik.gauss(src, s))
				lt = t
			# Interpolate the arrival time at the receiver
			grcv = bx.cart2cell(*rcv)
			atimes[t,r] = tmi.evaluate(*grcv, grad=False)

		if not rank and i == ipow:
			ipow <<= 1
			print 'Rank 0: Finished path %d of %d' % (i, share)

	# Make sure all participants have finished
	MPI.COMM_WORLD.Barrier()

	# Collect all arrival times
	atimes = MPI.COMM_WORLD.gather(atimes)

	if not rank:
		# Collapse individual arrival-time maps
		atimes = { (t, r): v for l in atimes for (t, r), v in l.iteritems() }
		savez_keymat(output, atimes)
