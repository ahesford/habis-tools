#!/usr/bin/env python

# Copyright (c) 2016 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os, sys, numpy as np

from itertools import izip

from mpi4py import MPI

from habis.habiconf import HabisConfigParser, HabisConfigError, matchfiles
from habis.formats import loadmatlist as ldmats, loadkeymat as ldkmat, savez_keymat

from pycwp import boxer, cutil


def usage(progname=None, fatal=True):
	if not progname: progname = os.path.basename(sys.argv[0])
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname
	sys.exit(int(fatal))


def getspdpaths(atfiles, elements, vclip, rank=0, size=1):
	'''
	Read and unify the arrival-time maps in the iterable atfiles, which
	provides file names of 2-key maps. Filter the map, atimes, to include
	only keys sorted(k for k in atimes.keys() if k[0] != k[1])[rank::size].
	For each (t, r) key in the filtered map, return the mappings

		segs = { (t, r): boxer.Segment3D(elements[t], elements[r]) }

	and

		spds = { (t, r): segs[t,r].length / atimes[t,r]
			if vclip[0] <= segs[t,r] / atimes[t,r] <= vclip[1] }.
	'''
	# Load all arrival-time maps and eliminate times not of interest
	atimes = ldmats(atfiles, nkeys=2)

	# Remove backscatter and keep a local share
	keys = sorted(k for k in atimes if k[0] != k[1])[rank::size]

	# Build the local collection of segments to trace
	segs = { k: boxer.Segment3D(*(elements[kv] for kv in k)) for k in keys }

	# Find average speeds, ignoring values outside of clipping range
	spds = { k: v for k, s in segs.iteritems()
			for v in (s.length / atimes[k],) if vclip[0] <= v <= vclip[1] }

	return segs, spds


def tracerEngine(config):
	'''
	Given lists of arrival-time maps, element positions, and lists of
	control points and triangles that define a target surface, all
	specified in the HabisConfigParser instance config, use a
	pycwp.boxer.Octree to identify points of intersections between the
	surface and each non-backscatter propagation path for which arrival
	times are defined. From these intersection points, deduce the fraction
	of each propagation path is spent inside the target, and determine an
	average interior sound speed from the arrival time and an assumed
	background sound speed.
	'''
	tsec = 'tracer'

	def _throw(msg, e, sec=tsec):
		raise HabisConfigError.fromException(err + ' in [%s]' % (sec,), e)

	# Find all arrival-time maps visible to this node
	try: timefiles = matchfiles(config.getlist(tsec, 'timefile'))
	except Exception as e: _throw('Configuration must specify timefile', e)

	# Find and load all element coordinates
	try: elements = ldmats(matchfiles(config.getlist(tsec, 'elements')), nkeys=1)
	except Exception as e: _throw('Configuration must specify elements', e)

	# Load the node and triangle configuration
	try: mesh = np.load(config.get(tsec, 'mesh'))
	except Exception as e: _throw('Configuration must specify mesh', e)

	try: levels = config.get(tsec, 'levels', mapper=int)
	except Exception as e: _throw('Configuration must specify levels', e)

	# Read the background sound speed or use water at 68F by default
	try: vbg = config.get('measurement', 'c', mapper=float, default=1.4823)
	except Exception as e: _throw('Optional c is invalid', e, 'measurement')

	try:
		# Pull a range of valid sound speeds for clipping
		vclip = tuple(config.getlist(tsec, 'vclip',
				mapper=float, default=(0., float('inf'))))
		if len(vclip) != 2:
			raise ValueError('Range must specify two elements')
	except Exception as e:
		_throw('Optional vclip is invalid', e)

	try: outfile = config.get(tsec, 'outfile')
	except Exception as e: _throw('Configuration must specify outfile', e)

	mpirank, mpisize = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

	MPI.COMM_WORLD.Barrier()

	# Convert the triangle node maps to triangle objects
	nodes, triangles = mesh['nodes'], mesh['triangles']
	triangles = [ boxer.Triangle3D([nodes[c] for c in v]) for v in triangles ]

	# Compute an overall bounding box and an Octree for the space
	rootbox = boxer.Box3D(*zip(*((min(j), max(j))
		for j in izip(*(nd for tr in triangles for nd in tr.nodes)))))
	otree = boxer.Octree(levels, rootbox)

	if not mpirank:
		print 'Scatterer has %d triangles, %d nodes' % (len(triangles), len(nodes))
		print 'Limiting sound-speed values to range %s' % (vclip,)
		print 'Creating Octree decomposition (%d levels)' % (otree.level,)

	# Classify triangles according to overlaps with boxes in tree
	def inbox(b, i): return triangles[i].overlaps(b)
	otree.addleaves(xrange(len(triangles)), inbox, True)

	MPI.COMM_WORLD.Barrier()

	if not mpirank:
		print 'Computing average speeds over propagation paths'

	# Compute the local share of average sound speeds
	segs, spds = getspdpaths(timefiles, elements, vclip, mpirank, mpisize)

	MPI.COMM_WORLD.Barrier()

	if not mpirank:
		print 'Approximate local share of paths is %d' % (len(spds),)

	# Track "misses", when the exterior fraction is (almost) unity
	misses = set()
	# For hits, note interior and exterior fractions
	hits = { }

	# Accumulate the total results for every segment
	results = { }

	for k, aspd in spds.iteritems():
		# The segment associated with an average speed
		s = segs[k]
		# Skip very small segments
		if cutil.almosteq(s.length, 0.0, 1e-6): continue

		# A predicate to match box intersections with this segment
		def bsect(b): return b.intersection(s)
		# A predicate to match triangle intersections; cache results
		trcache = { }
		def lsect(i):
			try: return trcache[i]
			except KeyError:
				v = triangles[i].intersection(s)
				trcache[i] = v
				return v

		# Find intersections between segment and surface
		isects = otree.search(bsect, lsect)

		# Sort the lengths and add endpoints to define all subsegments
		ilens = sorted([0.] + [v[0] for v in isects.itervalues()] + [s.length])

		# Track average and (undefined) interior speeds and intersection lengths
		results[k] = [aspd, float('nan')] + ilens

		# Regions starting at odd indices are interior, at even are exterior
		inlen = sum(v[1] - v[0] for v in izip(ilens[1::2], ilens[2::2]))
		exlen = sum(v[1] - v[0] for v in izip(ilens[0::2], ilens[1::2]))

		if not cutil.almosteq(inlen + exlen, s.length, 1e-6):
			print ('WARNING: inferred length %0.5g for '
				'segment %s disagrees with actual '
				'length %0.5g' % (inlen + exlen, k, s.length))

		# Compute the interior and exterior fractions
		infrac = inlen / s.length
		exfrac = exlen / s.length

		if cutil.almosteq(exfrac, 1.0, 1e-6): misses.add(k)
		else: hits[k] = (infrac, exfrac)

	MPI.COMM_WORLD.Barrier()

	if not mpirank: print 'Finished tracing segments'

	# Find the average background sound speed for target misses
	nmiss = MPI.COMM_WORLD.allreduce(len(misses))
	tmiss = MPI.COMM_WORLD.allreduce(sum(spds[k] for k in misses))

	if nmiss > 0:
		# Override the nominal background speed based on misses
		vbg = tmiss / float(nmiss)
		if not mpirank:
			print 'Inferred background speed %0.3f (%d paths)' % (vbg, nmiss)

	for k, (infrac, exfrac) in hits.iteritems():
		# Update the interior speed for hits
		results[k][1] = (spds[k] - (exfrac * vbg)) / infrac

	# Accumulate the on the root
	results = MPI.COMM_WORLD.gather(results)

	# Only the head node has remaining work
	if mpirank: return

	# Flatten the multiple dictionaries and save
	results = dict(kp for l in results for kp in l.iteritems())
	savez_keymat(outfile, results)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage()

	try:
		config = HabisConfigParser(sys.argv[1])
	except Exception as e:
		err = 'Unable to load configuration file %s' % sys.argv[1]
		raise HabisConfigError.fromException(err, e)

	try:
		tracerEngine(config)
	except Exception as e:
		rank = MPI.COMM_WORLD.rank
		print 'MPI rank %d: caught exception %s' % (rank, e)
		sys.stdout.flush()
		MPI.COMM_WORLD.Abort(1)
