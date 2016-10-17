#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os, sys, numpy as np
from numpy.linalg import norm

from collections import defaultdict

from pycwp import stats, cutil

from habis.habiconf import HabisConfigError, HabisConfigParser, matchfiles
from habis.formats import loadkeymat, loadmatlist

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
	except Exception as e:
		err = 'Configuration must specify c in [%s]' % msection
		raise HabisConfigError.fromException(err, e)

	# Read the element positions and backscatter arrival times
	elements = loadmatlist(eltfiles, nkeys=1)
	times = { k[0]: v
			for k, v in loadmatlist(timefiles, nkeys=2).iteritems()
			if k[0] == k[1] }

	reflectors = np.loadtxt(rflfile, ndmin=2)
	nrefl, nrdim = reflectors.shape

	if nrdim == 3:
		# Make sure the reflector specification includes speed
		reflectors = np.concatenate([reflectors, [[c]] * nrefl], axis=1)
	elif not 2 < nrdim < 6:
		raise ValueError('Reflector file must contain 3, 4 or 5 columns')

	# Identify elements for which coordinates and arrival times are known
	celts = sorted(set(times).intersection(elements))

	# Keep a list of control points for each reflectors
	cpts = [ ]

	for refl in reflectors:
		pos = refl[:3]
		lc = refl[3]

		# Convert all times to distances
		pdists = { }
		for el in celts:
			# Ray from reflector center to element
			dl = elements[el] - pos
			# Find distance to element and normalize ray
			ll = norm(dl)
			dl /= ll
			# Convert arrival time to distance from center along ray
			ds = ll - times[el] * lc / 2.0
			# Valid points will be inside segment or opposite reflector
			if ds <= ll:
				# Record distance and ray
				pdists[el] = (ds, dl)

		if olrange is not None:
			# Group distances according to olgroup
			pgrps = defaultdict(dict)
			for el, (ds, _) in pdists.iteritems():
				pgrps[int(el / olgroup)][el] = ds
			# Filter outliers from each group and add remaining points
			cpts.append([pos + pdists[el][0] * pdists[el][1]
				for pgrp in pgrps.itervalues()
				for el in stats.mask_outliers(pgrp, olrange)])
		else:
			# Add all control points if no outlier exclusion is needed
			cpts.append([pos + ds * dl for ds, dl in pdists.itervalues()])

	if len(cpts) == 1 and outfile.endswith('.txt'):
		# A single output can be stored in text format if desired
		np.savetxt(outfile, cpts[0])
	else:
		# Multiple targets or different extensions always demand npz format
		ndig = cutil.numdigits(len(cpts))
		tfmt = 'target%%0%dd' % (ndig,)
		np.savez(outfile, **{ tfmt % i: v for i, v in enumerate(cpts) })


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
