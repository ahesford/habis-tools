#!/usr/bin/env python

import os, sys, itertools, numpy as np
import multiprocessing

from itertools import izip

from pycwp import process

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
		timefiles = config.getlist('trilateration', 'timefile')
		if len(timefiles) < 1:
			err = 'Key timefile must contain at least one entry'
			raise HabisConfigError(err)
		inelements = config.getlist('trilateration', 'inelements')
		if len(inelements) != len(timefiles):
			err = 'Key inelements must contain as many entries as timefile'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify timefile and inelements in [trilateration]'
		raise HabisConfigError.fromException(err, e)

	# Grab the initial guess for reflector positions
	try:
		guessfile = config.get('trilateration', 'guessfile')
	except Exception as e:
		err = 'Configuration must specify guessfile in [trilateration]'
		raise HabisConfigError.fromException(err, e)

	# Grab the reflector output file
	try:
		outreflector = config.get('trilateration', 'outreflector')
	except Exception as e:
		err = 'Configuration must specify outreflector in [trilateration]'
		raise HabisConfigError.fromException(err, e)

	try:
		outelements = config.getlist('trilateration', 'outelements',
				failfunc=lambda: [''] * len(timefiles))
	except Exception as e:
		err = 'Invalid specification of optional outelements in [trilateration]'
		raise HabisConfigError.fromException(err, e)

	if len(outelements) != len(timefiles):
		err = 'Lists outelements and timefile must have equal length'
		raise HabisConfigError(err)

	try:
		# Grab the number of processes to use (optional)
		nproc = config.getint('general', 'nproc',
				failfunc=process.preferred_process_count)
	except Exception as e:
		err = 'Invalid specification of optional nproc in [general]'
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the sound speed and radius
		c = config.getfloat('trilateration', 'c')
		radius = config.getfloat('trilateration', 'radius')
	except Exception as e:
		err = 'Configuration must specify c and radius in [trilateration]'
		raise HabisConfigError.fromException(err, e)

	# Pull the element indices for each group file
	elements = [np.loadtxt(efile) for efile in inelements]
	# Pull the arrival times and convert surface times to center times
	times = [np.loadtxt(tfile) + ((2. * radius) / c) for tfile in timefiles]
	# Pull the reflector guess
	guess = np.loadtxt(guessfile)

	# Allocate a multiprocessing pool
	pool = multiprocessing.Pool(processes=nproc)

	# Concatenate times and element lists for reflector trilateration
	ctimes = np.concatenate(times, axis=0)
	celts = np.concatenate(elements, axis=0)

	# Compute the reflector positions in parallel
	# Use async calls to correctly handle keyboard interrupts
	result = pool.map_async(getreflpos,
			((t, celts, g, c) for t, g in izip(ctimes.T, guess)))
	while True:
		try:
			reflectors = result.get(5)
			break
		except multiprocessing.TimeoutError:
			pass

	# Save the reflector positions
	np.savetxt(outreflector, reflectors, fmt='%16.8f')

	for ctimes, celts, ofile in izip(times, elements, outelements):
		# Don't do trilateration if the result will not be saved
		if not len(ofile): continue

		# Create and save a refined estimate of the reflector locations
		pltri = trilateration.PlaneTrilateration(reflectors, c)
		relements = pltri.newton(ctimes, pos=celts)
		np.savetxt(ofile, relements, fmt='%16.8f')


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
