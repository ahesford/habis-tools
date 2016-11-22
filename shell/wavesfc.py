#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os, sys, numpy as np
from numpy.linalg import norm

from scipy.spatial import Delaunay

from collections import defaultdict

from pycwp import stats, cutil

from habis.habiconf import HabisConfigError, HabisConfigParser, matchfiles
from habis.formats import loadmatlist, savez_keymat, savetxt_keymat

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def wavesfcEngine(config):
	'''
	Use positions of elements and targets specified in the provided config,
	combined with a specified sound speed (to override per-reflector sound
	speeds), to compute control points that define a reflector surface.
	'''
	msection = 'measurement'
	wsection = 'wavesfc'
	try:
		# Try to grab the input and output files
		eltfiles = matchfiles(config.getlist(wsection, 'elements'))
		rflfile = config.get(wsection, 'reflectors')
		outfile = config.get(wsection, 'output')
	except Exception as e:
		err = 'Configuration must specify elements, reflectors and output in [%s]' % wsection
		raise HabisConfigError.fromException(err, e)

	try:
		# Try to read the input time files
		timefiles = matchfiles(config.getlist(wsection, 'timefile'))
	except Exception as e:
		err = 'Configuration must specify timefile in [%s]' % wsection
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the outlier range and group size
		olrange = config.get(wsection, 'olrange', mapper=float, default=None)
		olgroup = config.get(wsection, 'olgroup', mapper=int, default=64)
	except Exception as e:
		err = 'Invalid specification of optionals outrange or outgroup in [%s]' % wsection
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the sound speed
		c = config.get(msection, 'c', mapper=float)
		rad = config.get(msection, 'radius', mapper=float)
	except Exception as e:
		err = 'Configuration must specify c and radius in [%s]' % msection
		raise HabisConfigError.fromException(err, e)

	try:
		# Use the reflector radius for missing arrival-time values
		usemiss = config.get(wsection, 'usemiss', mapper=bool, default=False)
	except Eception as e:
		err = 'Invalid specification of optional usemiss in [%s]' % wsection
		raise HabisConfigError.fromException(err, e)

	# Read the element positions and backscatter arrival times
	elements = loadmatlist(eltfiles, nkeys=1)
	times = { k[0]: v
			for k, v in loadmatlist(timefiles, scalar=False, nkeys=2).iteritems()
			if k[0] == k[1] }

	reflectors = np.loadtxt(rflfile, ndmin=2)
	nrefl, nrdim = reflectors.shape

	# Make sure the reflector specification includes speed and radius
	if nrdim == 3:
		reflectors = np.concatenate([reflectors, [[c, rad]] * nrefl], axis=1)
	elif nrdim == 4:
		reflectors = np.concatenate([reflectors, [[rad]] * nrefl], axis=1)
	elif nrdim != 5:
		raise ValueError('Reflector file must contain 3, 4 or 5 columns')

	# Split the file name and extension
	fbase, fext = os.path.splitext(outfile)

	# Add a target specifier to outputs for multiple targets
	nrefl = len(reflectors)
	if nrefl == 1: idfmt = ''
	else: idfmt = '.Target{{0:0{width}d}}'.format(width=cutil.numdigits(nrefl))

	for ridx, refl in enumerate(reflectors):
		pos = refl[:3]
		lc, rad = refl[3:]

		# Convert all times to distances
		pdists = { }
		for el, elpos in elements.iteritems():
			# Ray from reflector center to element
			dl = elpos - pos
			# Find distance to element and normalize ray
			ll = norm(dl)
			dl /= ll

			try:
				# Convert arrival time to distance from center
				ds = ll - times[el][ridx] * lc / 2.0
			except (KeyError, IndexError, TypeError):
				# Use default radius or skip missing value
				if usemiss: ds = rad
				else: continue

			# Valid points will be inside segment or opposite reflector
			if ds <= ll:
				# Record distance and ray
				pdists[el] = (ds, dl)

		if olrange is not None:
			# Group distances according to olgroup
			pgrps = defaultdict(dict)
			for el, (ds, _) in pdists.iteritems():
				pgrps[int(el / olgroup)][el] = ds
			# Filter outliers from each group and flatten map
			pdists = { el: pdists[el] for pg in pgrps.itervalues()
					for el in stats.mask_outliers(pg, olrange) }

		# Sort remaining values, separate indices and control points
		cpel = sorted(pdists.iterkeys())
		cpts = np.array([ pos + pdists[el][0] * pdists[el][1] for el in cpel ])

		fname = fbase + idfmt.format(ridx) + fext

		# Project control points for tesselation
		zmin = np.min(cpts[:,-1])
		zmax = np.max(cpts[:,-1])

		if np.allclose(zmin, zmax):
			# For flat structures, just use x-y
			tris = Delaunay(cpts[:,:2])
		else:
			# Otherwise, project radially
			zdiff = zmax - zmin
			prctr = np.mean(cpts, axis=0)
			prctr[-1] = zmax + zdiff
			prref = np.array(prctr)
			prref[-1] = zmin

			l = cpts - prctr[np.newaxis,:]
			d = (prref - prctr)[-1] / l[:,-1]

			if np.any(np.isinf(d)) or np.any(np.isnan(d)):
				raise ValueError('Failure in projection for tesselation')

			ppts = prctr[np.newaxis,:] + d[:,np.newaxis] * l
			tris = Delaunay(ppts[:,:2])

		# Save the control points and triangles
		np.savez(fname, nodes=cpts, triangles=tris.simplices, elements=cpel)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Read the configuration file
	try:
		config = HabisConfigParser(sys.argv[1])
	except:
		print >> sys.stderr, 'ERROR: could not load configuration file %s' % sys.argv[1]
		usage(sys.argv[0])
		sys.exit(1)

	# Call the calculation engine
	wavesfcEngine(config)
