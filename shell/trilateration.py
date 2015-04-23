#!/usr/bin/env python

import os, sys, itertools, ConfigParser, numpy as np
import multiprocessing
import socket

from operator import mul

from habis import trilateration

# Define a new ConfigurationError exception
class ConfigurationError(Exception): pass

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def getreflpos(args):
	'''
	For args = (times, elemlocs, guess, c), with a list of N round-trip
	arrival times and an N-by-3 matrix elemlocs of reference element
	locatations, use Newton's method in
	habis.trilateration.PointTrilateration to determine, starting with the
	provided 3-dimensional guess position, the position of a zero-radius
	reflector in a medium with sound speed c.

	The return value is a 3-tuple specifying the final coordinates of the
	center.
	'''
	times, elemlocs, guess, c = args
	t = trilateration.PointTrilateration(elemlocs, c)
	# Offset the arrival times by the propagation time
	# from reflector surface to center and back
	return t.newton(times, pos=guess)


def geteltpos(times, reflocs, guess, c):
	'''
	Given an N-by-M matrix of round-trip arrival times and an M-by-3 matrix
	reflocs of reference element location, use Newton's method in
	habis.trilateration.PlaneTrilateration to determine, starting with the
	N-by-3 matrix of guess positions, the locations of N elements in a
	medium with sound speed c.

	The return value is an N-by-3 matrix specifying the final coordinates
	of the center.
	'''
	t = trilateration.PlaneTrilateration(reflocs, c)
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
		raise ConfigurationError('Configuration must specify timefile, guessfile, and facetfile in [trilateration]')

	try:
		# Grab the output file locations
		outreflector = config.get('trilateration', 'outreflector')
		outelements = config.get('trilateration', 'outelements')
	except ConfigParser.Error:
		raise ConfigurationError('Configuration must specify outreflector and outfacet in [trilateration]')

	try:
		# Grab the element range
		elementRange = [int(s) for s in config.get('trilateration', 'elementrange').split()]
	except:
		raise ConfigurationError('Configuration must specify elementrange in [trilateration]')

	try:
		# Grab the number of processes to use (optional)
		nproc = int(config.get('general', 'nproc'))
	except ConfigParser.NoOptionError:
		nproc = process.preferred_process_count()
	except:
		raise ConfigurationError('Invalid specification of process count in [general]')

	try:
		# Grab the sound speed and radius
		c = float(config.get('trilateration', 'c'))
		radius = float(config.get('trilateration', 'radius'))
	except:
		raise ConfigurationError('Configuration must specify sound-speed c and radius in [trilateration]')

	# Pull the element indices
	elements = np.loadtxt(facetfile)[range(*elementRange)]
	# Pull the arrival times and convert surface times to center times
	times = np.loadtxt(timefile) + ((2. * radius) / c)
	# Pull the reflector guess
	guess = np.loadtxt(guessfile)

	# Allocate a multiprocessing pool
	pool = multiprocessing.Pool(processes=nproc)

	# Compute the reflector positions in parallel
	result = pool.map_async(getreflpos,
			((t, elements, g, c) for t, g in zip(times.T, guess)))

	# Loop to check for completed results
	while True:
		try:
			reflectors = np.array(result.get(0.1))
			# Stop looping when the result is available
			break
		except multiprocessing.TimeoutError: pass

	# Save the reflector positions
	np.savetxt(outreflector, reflectors, fmt='%16.8f')

	# Create and save a refined estimate of the reflector locations
	relements = geteltpos(times, reflectors, elements, c)
	np.savetxt(outelements, relements, fmt='%16.8f')


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Read the configuration file
	config = ConfigParser.SafeConfigParser()
	if len(config.read(sys.argv[1])) == 0:
		print >> sys.stderr, 'ERROR: configuration file %s does not exist' % sys.argv[1]
		usage(sys.argv[0])
		sys.exit(1)

	# Call the calculation engine
	trilaterationEngine(config)
