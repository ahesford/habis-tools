#!/usr/bin/env python

import os, sys, itertools, ConfigParser, numpy as np
import multiprocessing

from habis import trilateration
from habis.habiconf import HabisConfigError, HabisConfigParser

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def getreflpos(args):
	'''
	For args = (times, elocs, guess, c), with a list of N round-trip
	arrival times and an N-by-3 matrix elocs of reference element
	locatations, use habis.trilateration.PointTrilateration to determine,
	starting with the provided 3-dimensional guess position, the position
	of a zero-radius reflector in a medium with sound speed c.

	The return value is the 3-dimensional position of the reflector.
	'''
	times, elemlocs, guess, c = args
	t = trilateration.PointTrilateration(elemlocs, c)
	return t.newton(times, pos=guess)


def trilaterationEngine(config):
	'''
	Use the PointTrilateration and PlaneTrilateration classes in
	habis.trilateration to determine, iteratively from a set of
	measurements of round-trip arrival times, the unknown positions of
	a set of reflectors followed by estimates of the positions of the
	hemisphere elements.
	'''
	try:
		# Try to grab the input files
		timefile = config.get('trilateration', 'timefile')
		guessfile = config.get('trilateration', 'guessfile')
		facetfile = config.get('trilateration', 'facetfile')
	except ConfigParser.Error:
		raise HabisConfigError('Configuration must specify timefile, guessfile, and facetfile in [trilateration]')

	try:
		# Grab the output file locations
		outreflector = config.get('trilateration', 'outreflector')
		outelements = config.get('trilateration', 'outelements')
	except ConfigParser.Error:
		raise HabisConfigError('Configuration must specify outreflector and outfacet in [trilateration]')

	try:
		# Grab the element range
		elements = config.getrange('trilateration', 'elements')
	except:
		raise HabisConfigError('Configuration must specify elements in [trilateration]')

	try:
		# Grab the number of processes to use (optional)
		nproc = config.getint('general', 'nproc')
	except ConfigParser.NoOptionError:
		nproc = process.preferred_process_count()
	except:
		raise HabisConfigError('Invalid specification of process count in [general]')

	try:
		# Grab the sound speed and radius
		c = config.getfloat('trilateration', 'c')
		radius = config.getfloat('trilateration', 'radius')
	except:
		raise HabisConfigError('Configuration must specify sound-speed c and radius in [trilateration]')

	# Pull the desired element indices
	elements = np.loadtxt(facetfile)[elements]
	# Pull the arrival times and convert surface times to center times
	times = np.loadtxt(timefile) + ((2. * radius) / c)
	# Pull the reflector guess
	guess = np.loadtxt(guessfile)

	# Allocate a multiprocessing pool
	pool = multiprocessing.Pool(processes=nproc)

	# Compute the reflector positions in parallel
	# Use async calls to correctly handle keyboard interrupts
	result = pool.map_async(getreflpos,
			((t, elements, g, c) for t, g in zip(times.T, guess)))
	while True:
		try:
			reflectors = result.get(5)
			break
		except multiprocessing.TimeoutError:
			pass

	# Save the reflector positions
	np.savetxt(outreflector, reflectors, fmt='%16.8f')

	# Create and save a refined estimate of the reflector locations
	pltri = trilateration.PlaneTrilateration(reflectors, c)
	relements = pltri.newton(times, pos=elements)
	np.savetxt(outelements, relements, fmt='%16.8f')


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
	trilaterationEngine(config)
