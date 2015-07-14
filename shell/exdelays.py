#!/usr/bin/env python

import os, sys, ConfigParser, numpy as np
from numpy.linalg import norm

from habis.habiconf import HabisConfigError, HabisConfigParser

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
		c = config.getfloat(msection, 'c')
		r = config.getfloat(msection, 'radius')
	except Exception as e: 
		err = 'Configuration must specify c and radius in [%s]' % msection
		raise HabisConfigError.fromException(err, e)

	# Read the element and reflector positions
	eltspos = np.concatenate([np.loadtxt(efile) for efile in eltfiles], axis=0)
	reflpos = np.loadtxt(rflfile)

	_, nedim = eltspos.shape
	_, nrdim = reflpos.shape
	if nedim != nrdim != nedim + 1:
		raise ValueError('Reflector and element dimensionalities are incompatible')

	# Determine the one-way distances between elements and reflector centers
	# Ignore an extra dimension (wave speed) in reflector coordinates
	dx = norm(eltspos[:,np.newaxis,:] - reflpos[np.newaxis,:,:nedim], axis=-1)
	# Convert distances to round-trip arrival times
	times = 2 * (dx - r) / c

	# Save the estimated arrival times
	np.savetxt(timefile, times, fmt='%16.8f')


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Read the configuration file
	try:
		config = HabisConfigParser.fromfile(sys.argv[1])
	except:
		print >> sys.stderr, 'ERROR: could not load configuration file %s' % sys.argv[1]
		usage(sys.argv[0])
		sys.exit(1)

	# Call the calculation engine
	exdelayEngine(config)
