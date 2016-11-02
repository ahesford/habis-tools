#!/usr/bin/env python

# Copyright (c) 2016 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os, sys, numpy as np
from numpy.linalg import norm

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr


from itertools import izip

from mpi4py import MPI

from habis.habiconf import HabisConfigParser, HabisConfigError, matchfiles
from habis.formats import loadmatlist as ldmats, loadkeymat as ldkmat, savez_keymat

from pycwp import boxer, cutil


def usage(progname=None, fatal=True):
	if not progname: progname = os.path.basename(sys.argv[0])
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname
	sys.exit(int(fatal))


def getatimes(atfiles, elements, vclip=None):
	'''
	Read and unify the arrival-time maps in the iterable atfiles, which
	provides file names of 2-key maps. Filter the map, atimes, to include
	only keys such that each index is a key in the mapping elements from
	indices to element locations, and the average speed (the propagation
	path length, computed from element locations, divided by the arrival
	time) falls between vclip[0] and vclip[1].

	Backscatter waveforms are included in the map but will not be filtered
	by vclip because the average sound speed is undefined for backscatter.
	'''
	# Load all arrival-time maps and eliminate times not of interest
	atimes = { }
	for (t, r), v in ldmats(atfiles, nkeys=2).iteritems():
		try: elt, elr = elements[t], elements[r]
		except KeyError: continue

		if t != r and vclip:
			aspd = norm(elt - elr) / v
			if aspd < vclip[0] or aspd > vclip[1]: continue

		atimes[t,r] = v

	return atimes


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

	# Find and load all propagation axes
	try: meshctr = np.loadtxt(config.get(tsec, 'meshctr'), ndmin=2)[0,:3]
	except Exception as e: _throw('Configuration must specify meshctr', e)

	# Load the node and triangle configuration
	try: mesh = np.load(config.get(tsec, 'mesh'))
	except Exception as e: _throw('Configuration must specify mesh', e)

	try: levels = config.get(tsec, 'levels', mapper=int)
	except Exception as e: _throw('Configuration must specify levels', e)

	# Read the background sound speed or use water at 68F by default
	try: vbg = config.get('measurement', 'c', mapper=float, default=1.4823)
	except Exception as e: _throw('Invalid optional c', e, 'measurement')

	try:
		# Pull a range of valid sound speeds for clipping
		vclip = config.getlist(tsec, 'vclip', mapper=float, default=None)
		if vclip:
			if len(vclip) != 2:
				raise ValueError('Range must specify two elements')
			if vclip[0] > vclip[1]:
				raise ValueError('Minimum value must not exceed maximum')
			vclip = tuple(vclip)

	except Exception as e:
		_throw('Invalid optional vclip', e)

	try: lsmr_opts = config.get(tsec, 'lsmr', mapper=dict, default={ })
	except Exception as e: _throw('Invalid optional lsmr', e)

	try: bimodal = config.get(tsec, 'bimodal', mapper=bool, default=True)
	except Exception as e: _throw('Invalid optional bimodal', e)

	try: pathfile = config.get(tsec, 'pathfile')
	except Exception as e: _throw('Configuration must specify pathfile', e)

	try: speedfile = config.get(tsec, 'speedfile')
	except Exception as e: _throw('Configuration must specify speedfile', e)

	WORLD = MPI.COMM_WORLD
	mpirank, mpisize = WORLD.rank, WORLD.size

	WORLD.Barrier()

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
	# Only build the tree for a local share of triangles
	otree.addleaves(xrange(mpirank, len(triangles), mpisize), inbox, True)

	WORLD.Barrier()

	if not mpirank: print 'Combining distributed Octree'

	# Merge leaves from all othe rranks
	for leaves in WORLD.allgather(otree.getleaves()):
		otree.mergeleaves(leaves)

	# Prune the tree and print some statistics
	otree.prune()

	WORLD.Barrier()

	if not mpirank: print 'Reading and gathering arrival times'

	# Read local arrival times, eliminate out-of-bounds values
	atimes = getatimes(timefiles, elements, vclip)
	# Gather all arrival times on all ranks
	atimes = dict(kp for atl in WORLD.allgather(atimes) for kp in atl.iteritems())
	# Pull a local share of the arrival times, sorted
	atimes = { k: atimes[k] for k in sorted(atimes)[mpirank::mpisize] }

	if not mpirank: print 'Approximate local share of paths is %d' % (len(atimes),)

	# Build the segment list for the local arrival times
	segs = { }
	for t, r in atimes:
		if t != r:
			segs[t,r] = boxer.Segment3D(elements[t], elements[r])
			continue
		# For backscatter, make segment length encompass volume
		epos = elements[t]
		segs[t,t] = boxer.Segment3D(epos, meshctr)

	WORLD.Barrier()

	# Accumulate the total results for every segment
	results = { }

	# Find the portion of each path in the interior and exterior of the volume
	for k, seg in segs.iteritems():
		# Skip very small segments
		if cutil.almosteq(seg.length, 0.0, 1e-6): continue

		# A predicate to match segment-box intersections
		def bsect(b): return b.intersection(seg)

		# For speed, cache segment-triangle intersections in predicate
		trcache = { }
		def tsect(i):
			try:
				return trcache[i]
			except KeyError:
				v = triangles[i].intersection(seg)
				trcache[i] = v
				return v

		# Find intersections between segment and surface
		isects = otree.search(bsect, tsect)

		# For backscatter, exterior path is to first intersection and back
		if k[0] == k[1]:
			# Backscatter is nonsense if there is no intersection
			if not len(isects): continue
			# Record exterior path length
			exl = 2.0 * min(v[0] for v in isects.itervalues())
			results[k] = (exl, 0.0, atimes[k])
			continue

		# Sort the lengths and add endpoints to define all subsegments
		ilens = sorted([0.] + [v[0] for v in isects.itervalues()] + [seg.length])

		# Odd intersections count means segment starts or ends in interior
		if len(ilens) % 2:
			print ('WARNING: (t,r) segment %s intersects '
					'volume an odd number of times' % (k,))

		# Regions starting at odd indices are interior, at even are exterior
		inl = sum(v[1] - v[0] for v in izip(ilens[1::2], ilens[2::2]))
		exl = sum(v[1] - v[0] for v in izip(ilens[0::2], ilens[1::2]))
		ttl = inl + exl

		if not cutil.almosteq(ttl, seg.length, 1e-6):
			print ('WARNING: (t,r) segment %s inferred length %0.5g, '
					'actual length %0.5g' % (ttl, k, seg.length))

		results[k] = (exl, inl, atimes[k])

	WORLD.Barrier()

	if not mpirank: print 'Finished tracing segments'

	# Accumulate all results on the root process
	results = WORLD.gather(results)

	# Only the root has any work left
	if mpirank: return

	# Combine the gathered result dictionaries and save the output
	results = { k: v for rs in results for k, v in rs.iteritems() }
	savez_keymat(pathfile, results)

	# Grab the keys to define a sort order
	keys = sorted(results)

	# Build the linear system to invert for speeds
	if bimodal:
		A = np.array([results[k][:2] for k in keys])
	else:
		# In a multi-medium model, build a sparse matrix
		# Track the keys with interior speeds
		ikeys = []
		data = []
		rowcol = []
		for i, k in enumerate(keys):
			exl, inl = results[k][:2]
			# Exterior speed in the first column
			rowcol.append((i, 0))
			data.append(exl)

			if abs(inl > 1e-6):
				# Unique interior speed in its own column
				ikeys.append(k)
				rowcol.append((i, len(ikeys)))
				data.append(inl)

		A = csr_matrix((data, zip(*rowcol)))

	b = np.array([results[k][2] for k in keys])

	x, istop, itn = lsmr(A, b, **lsmr_opts)[:3]
	if istop != 1: print 'LSMR terminated with istop', istop

	# Invert the inverted speeds
	x = tuple(1. / xv if abs(xv) > 1e-6 else 0.0 for xv in x)

	if bimodal:
		print 'Recovered exterior speed: %0.5g' % (x[0],)
		print 'Recovered interior speed: %0.5g' % (x[1],)
		np.savetxt(speedfile, x)
	else:
		savez_keymat(speedfile, { k: (x[0], v) for k, v in izip(ikeys,x[1:]) })

	print 'LSMR iterations: %d' % (itn,)


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
