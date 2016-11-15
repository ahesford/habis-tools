#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os, sys, ConfigParser, numpy as np
from numpy.linalg import norm

from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.formats import loadkeymat, savez_keymat

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def exdelayEngine(config):
	'''
	Use positions of elements and reflectors specified in the provided
	config, combined with a specified sound speed and reflector radius, to
	estimate the round-trip arrival times from every element to the
	reflector and back.
	'''
	msection = 'measurement'
	esection = 'exdelays'
	try:
		# Try to grab the input and output files
		eltfiles = config.getlist(esection, 'elements')
		rflfile = config.get(esection, 'reflectors')
		timefile = config.get(esection, 'timefile')
	except Exception as e:
		err = 'Configuration must specify elements, reflectors and timefile in [%s]' % esection
		raise HabisConfigError.fromException(err, e)

	# Grab the sound speed and reflector radius
	try:
		c = config.get(msection, 'c', mapper=float)
		r = config.get(msection, 'radius', mapper=float)
	except Exception as e: 
		err = 'Configuration must specify c and radius in [%s]' % msection
		raise HabisConfigError.fromException(err, e)

	try:
		# Read an optional global time offset
		offset = config.get(esection, 'offset', mapper=float, default=0.)
	except Exception as e:
		err = 'Invalid optional offset in [%s]' % esection
		raise HabisConfigError.fromException(err, e)

	# Read the element and reflector positions
	eltspos = dict(kp for efile in eltfiles
			for kp in loadkeymat(efile).iteritems())
	reflpos = np.loadtxt(rflfile, ndmin=2)
	nrefl, nrdim = reflpos.shape

	times = {}
	for elt, epos in eltspos.iteritems():
		nedim = len(epos)
		if not nedim <= nrdim <= nedim + 2:
			raise ValueError('Incompatible reflector and element dimensionalities')
		# Determine one-way distances between element and reflector centers
		dx = norm(epos[np.newaxis,:] - reflpos[:,:nedim], axis=-1)
		# Use encoded wave speed if possible, otherwise use global speed
		try: lc = reflpos[:,nedim]
		except IndexError: lc = c
		# Use encoded radius if possible, otherwise use global radius
		try: lr = reflpos[:,nedim+1]
		except IndexError: lr = r
		# Convert distances to round-trip arrival times
		times[elt,elt] = 2 * (dx - lr) / lc + offset

	# Save the estimated arrival times
	savez_keymat(timefile, times)


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
	exdelayEngine(config)
