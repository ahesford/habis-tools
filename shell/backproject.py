#!/usr/bin/env python

# Copyright (c) 2016 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os, sys, numpy as np

from numpy.random import rand

from mpi4py import MPI

from habis.habiconf import HabisConfigParser, HabisConfigError, matchfiles
from habis.formats import loadmatlist as ldmats, loadkeymat as ldkmat

from pycwp import boxer


def usage(progname=None, fatal=True):
	if not progname: progname = os.path.basename(sys.argv[0])
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname
	sys.exit(int(fatal))


def getspdpaths(atfiles, keylist, elements, vclip):
	'''
	Read each of the arrival-time maps encoded as 2-key maps in the
	sequence atfiles, pulling all arrival times with keys (t, r) such that
	(t, r) is an entry in the collection keylist.

	For the valid keys, prepare and return mappings

		segs = { (t, r): boxer.Segment3D(elements[t], elements[r]) }

	and

		spds = { (t, r): segs[t,r].length / atimes[t,r]
			if vclip[0] <= segs[t,r] / atimes[t,r] <= vclip[1] }

	for each (t, r) in the pulled arrival-time map atimes.
	'''
	# Load all arrival-time maps and eliminate times not of interest
	keyset = set((t, r) for t, r in keylist)
	atimes = { k: v for f in atfiles
			for k, v in ldkmat(f, nkeys=2).iteritems() if k in keyset }

	# Build the local collection of segments to trace
	segs = { (t, r): boxer.Segment3D(elements[t], elements[r])
			for t, r in atimes.iterkeys() }

	# Find average speeds, ignoring values outside of clipping range
	spds = { k: v for k in atimes.iterkeys()
				for v in [segs[k].length / atimes[k]]
					if vclip[0] <= v <= vclip[1] }

	return segs, spds


def makethresh(box, center, radius, fuzz=0.):
	'''
	Given a pycwp.boxer.Box3D instance box, a spherical center of the form
	center = (x, y, z) in world coordinates, a spherical radius and an
	optional fuzz half-width (each in world coordinates), return a
	distribution of probabilities (over the grid specified in box) that
	each pixel is exterior to the sphere.

	If fuzz is nonzero, the probability will ramp linearly from 0 inside a
	concentric sphere with radius (radius - fuzz) to 1 outside a concentric
	sphere with radius (radius + fuzz).
	'''
	if fuzz < 0:
		raise ValueError('Radial fuzz must be nonnegative')

	# Build an open grid of pixel steps in world coordinates
	dx, dy, dz = box.cell
	lx, ly, lz = box.lo
	hx, hy, hz = box.hi

	xg, yg, zg = np.ogrid[lx:hx:dx, ly:hy:dy, lz:hz:dz]
	xc, yc, zc = center

	rdsq = (xg - xc)**2 + (yg - yc)**2 + (zg - zc)**2

	if np.allclose(fuzz, 0.):
		thresh = (rdsq >= radius**2).astype(float)
	else:
		rmsq = (radius - fuzz)**2
		rscale = (radius + fuzz)**2 - rmsq
		thresh = np.clip((rdsq - rmsq) / rscale, 0., 1.)

	return thresh


def backprojectionEngine(config):
	'''
	Given lists of arrival-time maps, element positions, propagation
	path maps, and a two-medium probability distribution specified in the
	HabisConfigParser instance config, perform raymarching to trace the
	propagation paths through the distribution and determine the unknown
	internal sound speed from an assumed background sound speed and average
	speeds inferred from arrival times and propagation path lengths.

	A quasi-backprojection method is then used to distribute the sound
	speeds across the propagation paths. When multiple paths trace through
	a given image pixel in the reconstruction volume, the associated sound
	speeds for each path are averaged.
	'''
	bsec = 'backprojection'

	try:
		# Find all arrival-time maps visible to this node
		timefiles = matchfiles(config.getlist(bsec, 'timefile'))
	except Exception as e:
		err = 'Configuration must specify timefile in [%s]' % bsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Find and load all element coordinates
		elements = ldmats(matchfiles(config.getlist(bsec, 'elements')), nkeys=1)
	except Exception as e:
		err = 'Configuration must specify elements in [%s]' % bsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Find the propagation maps
		prfiles = matchfiles(config.getlist(bsec, 'propmap'))
		# Convert the maps into a sorted list of (t, r) pairs
		# Remove pairs that cannot be located in the element list
		prlist = [ ]
		for prf in prfiles:
			prmap = ldkmat(prf, nkeys=1, scalar=False)
			for r, tl in prmap.iteritems():
				prlist.extend((t, r) for t in set(tl)
						if t in elements and r in elements)
		prlist.sort()
	except Exception as e:
		err = 'Configuration must specify propmap in [%s]' % bsec
		raise HabisConfigError.fromException(err, e)

	try:
		# The grid must be a mapping with 'lo', 'hi', and 'ncell' keys
		grid = config.get(bsec, 'grid')
		box = boxer.Box3D(grid.pop('lo'), grid.pop('hi'))
		box.ncell = grid.pop('ncell')
		if grid:
			badkey = next(iter(grid))
			raise ValueError("Unrecognized key '%s' in grid mapping" % badkey)
	except Exception as e:
		err = 'Configuration must specify grid in [%s]' % bsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Read the pixel probability specification as a mapping or filename
		pixdist = config.get(bsec, 'pixdist')
		if isinstance(pixdist, basestring):
			# Treat the pixdist specification as a file name
			pixdist = np.load(pixdist)
			# Try to find an image in the loaded object
			if isinstance(pixdist, np.ndarray):
				# The file was an npy file describing a single array
				thresh = pixdist
			else:
				with pixdist:
					keys = pixdist.keys()
					# An npz file must have exactly one key,
					# or contain a 'pixdist' array
					if len(keys) > 1:
						thresh = pixdist['pixdist']
					else:
						thresh = pixdist[keys[0]]
		else:
			# Treat pixdist as kwargs to mkethresh
			thresh = makethresh(box, **pixdist)
		if tuple(thresh.shape) != box.ncell:
			raise ValueError('Pixel probability distribution does not fit image grid')
	except Exception as e:
		err = 'Configuration must specify pixdist in [%s]' % bsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Read the background sound speed or use water at 68F by default
		vbg = config.get('measurement', 'c', mapper=float, default=1.4823)
	except Exception as e:
		err = 'Invalid specifiction of optional c in [measurement]'
		raise HabisConfigError.fromException(err, e)

	try:
		# Pull a range of valid sound speeds for clipping
		vclip = tuple(config.getlist(bsec, 'vclip',
				mapper=float, default=(0., float('inf'))))
		if len(vclip) != 2:
			raise ValueError('Sound-speed clip range must specify two elements')
	except Exception as e:
		err = 'Invalid specification of optional vclip in [%s]' % bsec
		raise HabisConfigError.fromException(err, e)

	try:
		outfile = config.get(bsec, 'outfile')
	except Exception as e:
		err = 'Configuration must specify outfile in [%s]' % bsec
		raise HabisConfigError.fromException(err, e)

	mpirank, mpisize = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

	MPI.COMM_WORLD.Barrier()

	# Find the local share of arrival-time keys
	nkeys = len(prlist)
	share, srem = nkeys / mpisize, nkeys % mpisize
	start = mpirank * share + min(mpirank, srem)
	if mpirank < srem: share += 1

	if not mpirank:
		print 'Backprojection of sound-speed values, range %s' % (vclip,)
		print 'Image extent: %s x %s' % (box.lo, box.hi)
		print 'Image size: %s pixels' % (box.ncell,)
		print '%d sound-speed samples over %d processes (process share %d)' % (nkeys, mpisize, share)

	# Load all arrival-time maps and eliminate times not of interest
	segs, spds = getspdpaths(timefiles, prlist[start:start+share], elements, vclip)

	img = np.zeros(box.ncell, dtype=float)
	counts = np.zeros(box.ncell, dtype=float)

	for k, aspd in spds.iteritems():
		# Walk the segment to identify intersecting pixels
		# and probabilistically label each "interior" (0) or "exterior" (1)
		s = segs[k]
		pxv = [k + v + (int(rand() < thresh[k]),)
				for k, v in box.raymarcher(s).iteritems()]

		# Compute the "interior" and "exterior" fractions of the line
		pai, pbi = 0., 0.
		for pt in pxv:
			st, ed, md = pt[3:]
			length = abs(ed - st)
			pai += length * md
			pbi += length * (1 - md)

		pai /= float(s.length)
		pbi /= float(s.length)

		# Determine (if possible) the necessary interior sound speed
		vi = vbg if np.allclose(pbi, 0.) else ((aspd - vbg * pai) / float(pbi))
		vmap = [np.clip(vi, vclip[0], vclip[1]), vbg]

		# Add contributions of this segment to the image pixels
		for l, m, n, st, ed, md in pxv:
			length = abs(ed - st)
			counts[l,m,n] += length
			img[l,m,n] += length * vmap[md]

	MPI.COMM_WORLD.Barrier()

	# Accumulate the counts and image contributions on the head node
	img = MPI.COMM_WORLD.reduce(img, op=MPI.SUM)
	counts = MPI.COMM_WORLD.reduce(counts, op=MPI.SUM)

	# Only the head node has remaining work
	if mpirank: return

	# Average the image contributions and save the average and counts
	img /= counts
	np.savez(outfile, image=img, counts=counts)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage()

	try:
		config = HabisConfigParser(sys.argv[1])
	except Exception as e:
		err = 'Unable to load configuration file %s' % sys.argv[1]
		raise HabisConfigError.fromException(err, e)

	try:
		backprojectionEngine(config)
	except Exception as e:
		rank = MPI.COMM_WORLD.rank
		print 'MPI rank %d: caught exception %s' % (rank, e)
		sys.stdout.flush()
		MPI.COMM_WORLD.Abort(1)
