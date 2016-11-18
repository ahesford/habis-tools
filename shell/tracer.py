#!/usr/bin/env python

# Copyright (c) 2016 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os, sys, numpy as np
from numpy.linalg import norm

import hashlib

from itertools import izip

from mpi4py import MPI

from habis.habiconf import HabisConfigParser, HabisConfigError, matchfiles
from habis.formats import loadmatlist as ldmats, loadkeymat as ldkmat, savez_keymat

from pycwp import boxer, cutil


def makedtypes(nidx=2, nvals=3, names=None):
	'''
	Create and return a Numpy dtype and corresponding MPI structure data
	type consists of nidx 64-bit long integers followed by nvals 64-bit
	floats. The Numpy dtype is the first return value.

	If names is provided, it must be a string of length (nidx + nvals), or
	a sequence of (nidx + nvals) strings, that will be used to name the
	fields of the Numpy dtype; if a single string is provided, the fields
	will be given names corresponding to the characters within.

	If names is not provided, the dtype fields will be anonymous (i.e.,
	they will be named 'f0', 'f1', etc.).
	'''
	# Create data types to store and share segment intersection results
	dt = np.dtype(','.join(['<i8']*nidx + ['<f8']*nvals))
	if names: dt.names = names
	offsets = [dt.fields[n][1] for n in dt.names]
	mtypes = [MPI.LONG]*nidx + [MPI.DOUBLE]*nvals
	mpt = MPI.Datatype.Create_struct([1]*len(mtypes), offsets, mtypes)
	mpt.Commit()
	return dt, mpt


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


def gatherleaves(otree, comm=None):
	'''
	Gather on MPI communicator comm (or MPI.COMM_WORLD if comm is None) the
	leaves from Octree objects on all ranks and merge into the Octree
	otree. The leaves of otree will be sent by this rank.

	Nothing is returned.
	'''
	if comm is None: comm = MPI.COMM_WORLD
	for leaves in comm.allgather(otree.getleaves()):
		otree.mergeleaves(leaves)


def gathersegments(segs, mptype, comm=None, root=0):
	'''
	Gather, on the rank such that comm.Get_rank() == root for the
	communicator comm (MPI.COMM_WORLD if comm is None), all records in the
	rank-1 structured Numpy array segs with a Numpy dtype compatible with
	the MPI data type mptype.
	'''
	if comm is None: comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	# Gather the count from each rank
	nsegs = len(segs)
	counts = comm.gather(nsegs)

	# Allocate space on the root for all records; other ranks don't care
	if rank == root:
		combined = np.empty((sum(counts),), dtype=segs.dtype)
		# Compute the offsets for data received from each rank
		displs = [0]
		for ct in counts[:-1]: displs.append(displs[-1] + ct)
	else:
		combined, displs = None, None

	comm.Gatherv([segs, nsegs, mptype], [combined, counts, displs, mptype])
	return combined


def makerankmap(filemap, comm=None):
	'''
	Given filemap, a rank-local map from SHA-512 sums to names of files
	with those sums, gather the maps on all ranks in comm (MPI.COMM_WORLD
	if comm is None) and produce a composite map from SHA-512 sums to a
	list of ranks for which the sum is a key in its filemap.
	'''
	rankmap = { }
	if comm is None: comm = MPI.COMM_WORLD
	for rank, csums in enumerate(comm.allgather(filemap.keys())):
		for cs in csums:
			try: rankmap[cs].append(rank)
			except KeyError: rankmap[cs] = [rank]
	return rankmap


def usage(progname=None, fatal=True):
	if not progname: progname = os.path.basename(sys.argv[0])
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname
	sys.exit(int(fatal))


def getatimes(atfile, elements, column=0, vclip=None, start=0, stride=1):
	'''
	Read the 2-key arrival-time map with name atfile and filter the map to
	include only keys such that each index is a key in the mapping elements
	from indices to element locations, and the average speed (the propagation
	path length, computed from element locations, divided by the arrival
	time) falls between vclip[0] and vclip[1]. The column argument
	specifies which index in multi-value arrival-time maps should be selected.

	Backscatter waveforms are included in the map but will not be filtered
	by vclip because the average sound speed is undefined for backscatter.

	Only every stride-th *valid* record, starting with the start-th record,
	is retained.
	'''
	if not 0 <= start < stride:
		raise ValueError('Index start must be at least zero and less than stride')

	# Load the map, eliminate invalid elemenets, and keep the right portion
	atimes = { }
	idx = 0

	for (t, r), v in ldkmat(atfile, nkeys=2, scalar=False).iteritems():
		try: elt, elr = elements[t], elements[r]
		except KeyError: continue

		time = v[column]

		if t != r and vclip:
			aspd = norm(elt - elr) / time
			if aspd < vclip[0] or aspd > vclip[1]: continue

		# Keep every stride-th valid record
		if idx % stride == start: atimes[t,r] = time
		# Increment the valid record count
		idx += 1

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
		raise HabisConfigError.fromException(msg + ' in [%s]' % (sec,), e)

	# Find all arrival-time maps visible to this node
	try: timefiles = matchfiles(config.getlist(tsec, 'timefile'))
	except Exception as e: _throw('Configuration must specify timefile', e)

	try: targidx = config.get(tsec, 'targidx', mapper=int, default=0)
	except Exception as e: _throw('Invalid optional targidx', e)

	# Find and load all element coordinates
	try: elements = ldmats(matchfiles(config.getlist(tsec, 'elements')), nkeys=1)
	except Exception as e: _throw('Configuration must specify elements', e)

	# Find and load all propagation axes
	try: meshctr = np.loadtxt(config.get(tsec, 'meshctr'), ndmin=2)[targidx,:3]
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

	try: bimodal = config.get(tsec, 'bimodal', mapper=bool, default=True)
	except Exception as e: _throw('Invalid optional bimodal', e)

	try: fixbg = config.get(tsec, 'fixbg', mapper=bool, default=False)
	except Exception as e: _throw('Invalid optional fixbg', e)

	try: pathsave = config.get(tsec, 'pathsave', mapper=bool, default=False)
	except Exception as e: _throw('Invalid optional pathsave', e)

	try: output = config.get(tsec, 'output')
	except Exception as e: _throw('Configuration must specify output', e)

	try: epsilon = config.get(tsec, 'epsilon', default=1e-3)
	except Exception as e: _throw('Invalid optional epsilon', e)

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

	# Merge trees and prune
	gatherleaves(otree)
	otree.prune()

	WORLD.Barrier()

	if not mpirank: print 'Reading and gathering arrival times'

	# Map SHA-512 sums to file names and to lists of nodes with access each file
	timefiles = { sha512(t): t for t in timefiles }
	rankmap = makerankmap(timefiles)

	# Collect the arrival times from local shares of all local maps
	atimes = { }
	for cs, tfile in timefiles.iteritems():
		# Find the number of ranks sharing this file and index into it
		try:
			stride = len(rankmap[cs])
			start = rankmap[cs].index(mpirank)
		except (KeyError, ValueError):
			raise ValueError('Unable to determine local share of file' % (tfile,))
		atimes.update(getatimes(tfile, elements, targidx, vclip, start, stride))

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
	dt, mpt = makedtypes(2, 3, ('tx', 'rx', 'exlen', 'inlen', 'atime'))
	results = np.empty((len(segs),), dtype=dt)
	nres = 0
	# Track the number of odd-intersection paths that are skipped
	nodds = 0

	# Find the portion of each path in the interior and exterior of the volume
	for k, seg in segs.iteritems():
		# Skip very small segments
		if cutil.almosteq(seg.length, 0.0, epsilon): continue

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

		if k[0] == k[1]:
			# Backscatter: nonsense if there is no surface intersection
			if not len(isects): continue
			# Exterior path is to first hit and back, no interior
			exl = 2.0 * min(v[0] for v in isects.itervalues())
			inl = 0.0
		else:
			# Through transmissions: add endpoints to define all regions
			ilens = sorted([0., seg.length] +
					[v[0] for v in isects.itervalues()])

			# Odd count: segment starts or ends in interior
			if len(ilens) % 2:
				nodds += 1
				continue

			# Interior regions start at odd indices, exterior on evens
			inl = sum(v[1] - v[0] for v in izip(ilens[1::2], ilens[2::2]))
			exl = sum(v[1] - v[0] for v in izip(ilens[0::2], ilens[1::2]))

			# Check total length against segment length
			if not cutil.almosteq(inl + exl, seg.length, epsilon):
				print ('WARNING: (t,r) segment %s inferred '
					'and actual lengths differ: '
					'%0.5g != %0.5g' % (ttl, k, seg.length))

		# In fixed-exterior mode, ignore all-exterior segments
		if fixbg and abs(inl) <= epsilon * abs(exl): continue

		# Capture the arrival time for this segment
		atime = atimes[k]

		# Check speeds for sanity, if limits have been specified
		if vclip:
			if abs(inl) <= epsilon * abs(exl):
				# Propagation is all "exterior"
				vx = exl / atime
				if not vclip[0] <= vx <= vclip[1]: continue
			else:
				# Assume exterior speed is nominal value
				tex = exl / vbg
				if cutil.almosteq(atime, tex, epsilon): continue
				vi = inl / (atime - tex)
				if not vclip[0] <= vi <= vclip[1]: continue

		results[nres] = k + (exl, inl, atime)
		nres += 1

	# Accumulate the list of skipped segments on the root
	nodds = WORLD.reduce(nodds)
	if mpirank and nodds:
		print 'Skipped %d paths with an odd intersections count' % (nodds,)

	WORLD.Barrier()

	# Accumulate all results on the root
	results = gathersegments(results[:nres], mpt)
	# No need for the MPI type anymore
	mpt.Free()

	# Only the root has any work left
	if mpirank: return

	print 'Finished tracing; attempting to determine speeds'

	# Split tracing results into "hits" and "misses"
	hits = np.abs(results['inlen']) > epsilon * np.abs(results['exlen'])
	misses = results[np.logical_not(hits)]
	hits = results[hits]

	if not fixbg and len(misses):
		# Update the exterior speed, if desired and possible
		t = misses['atime']
		x = misses['exlen']
		vbg = np.dot(t, x) / np.dot(t, t)
		
	print 'Exterior speed: %0.5g (%d paths)' % (vbg, len(misses))

	# Offset arrival times by contribution from exterior speed
	rhs = hits['atime'] - hits['exlen'] / vbg

	if bimodal:
		# Bimodal matrix is a single column
		x = (np.dot(rhs, hits['inlen']) / np.dot(rhs, rhs), )
	else:
		# Multimodal matrix is diagonal
		x = tuple((hv / rv) if abs(rv) > epsilon else 0.
				for hv, rv in izip(hits['inlen'], rhs))

	# If Npz output will be used, always save the exterior speed
	szargs = { 'exspd': (vbg,) }

	if bimodal:
		print 'Interior speed: %0.5g (%d paths)' % (x[0], len(hits))
		if not pathsave:
			# Write a simple text file if paths aren't saved
			with open(output, 'wb') as f:
				f.write('# Exterior sound speed\n')
				f.write('%0.18g\n' % (vbg,))
				f.write('# Interior sound speed\n')
				f.write('%0.18g\n' % (x[0],))
			return
		# Store the interior speed is a single value (as a 1-element array)
		szargs['inspd'] = x
	else:
		# Print some useful stats on the recovered values
		stats = np.mean(x), np.median(x), np.std(x), len(x)
		print ('Interior speed: mean %0.5g, '
				'median %0.5g, std %0.5g (%d paths)' % stats)

		# Store the tx, rx, and speed values in multipath mode
		ist = [('tx', '<i8'), ('rx', '<i8'), ('inspd', '<f8')]
		ispds = np.empty(hits.shape, dtype=ist)
		ispds['tx'] = hits['tx']
		ispds['rx'] = hits['rx']
		ispds['inspd'] = x
		szargs['inspd'] = ispds

	if pathsave:
		# Store the hits and misses as paths only, if they exist
		if len(hits): szargs['hits'] = hits
		if len(misses): szargs['misses'] = misses

	# Save the output
	print 'Saving results'
	np.savez(output, **szargs)


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
