#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os, sys, ConfigParser, numpy as np
from numpy.linalg import norm

from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.formats import savekeymat, loadkeymat

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

	# Read the element and reflector positions
	eltspos = dict(kp for efile in eltfiles
			for kp in loadkeymat(efile).iteritems())
	reflpos = np.loadtxt(rflfile)
	nrefl, nrdim = reflpos.shape

	times = {}
	for elt, epos in eltspos.iteritems():
		nedim = len(epos)
		if nedim != nrdim != nedim + 1:
			raise ValueError('Reflector and element dimensionalities are incompatible')
		# Determine one-way distances between element and reflector centers
		dx = norm(epos[np.newaxis,:] - reflpos[:,:nedim], axis=-1)
		# Use encoded sound speeds if possible, otherwise use global speed
		try: lc = reflpos[:,nedim]
		except IndexError: lc = c
		# Convert distances to round-trip arrival times
		times[elt] = 2 * (dx - r) / lc

	# Save the estimated arrival times
	savekeymat(timefile, times, fmt=['%d'] + ['%16.8f']*nrefl)


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
