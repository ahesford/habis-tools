#!/usr/bin/env python

import os, sys, itertools, numpy as np
import multiprocessing

from itertools import izip

from pycwp import process, cutil

from habis import trilateration
from habis.habiconf import HabisConfigError, HabisConfigParser

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def getreflpos(args):
	'''
	For args = (times, elocs, guess, rad, c, varc), with a list of N
	round-trip arrival times and an N-by-3 matrix elocs of reference
	element locatations, use habis.trilateration.MultiPointTrilateration to
	determine, starting with the provided 3-dimensional guess position, the
	position of a reflector with radius rad in a medium with sound speed c.

	If varc is True, the MultiPointTrilateration is allowed to recover
	variable sound speed in addition to element position.

	The return value is a 4-dimensional vector in which the first three
	dimensions are the element position and the last dimension is the
	recovered sound speed. If varc is False, the fourth dimension just
	copies the input parameter c.
	'''
	times, elemlocs, guess, rad, c, varc = args
	t = trilateration.MultiPointTrilateration(elemlocs, rad, c)
	rval = t.newton(times, pos=guess, varc=varc)
	# Unpack the recovered solution
	pos, c = rval if varc else (rval, c)
	# Expand the position to include sound speed
	pos = np.concatenate([pos, [[c]] * pos.shape[0]], axis=1)
	return pos.squeeze()


def trilaterationEngine(config):
	'''
	Use the MultiPointTrilateration and PlaneTrilateration classes in
	habis.trilateration to determine, iteratively from a set of
	measurements of round-trip arrival times, the unknown positions of
	a set of reflectors followed by estimates of the positions of the
	hemisphere elements.
	'''
	tsection = 'trilateration'
	try:
		# Try to grab the input files
		timefiles = config.getlist(tsection, 'timefile')
		if len(timefiles) < 1:
			err = 'Key timefile must contain at least one entry'
			raise HabisConfigError(err)
		inelements = config.getlist(tsection, 'inelements')
		if len(inelements) != len(timefiles):
			err = 'Key inelements must contain as many entries as timefile'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify timefile and inelements in [%s]' % tsection
		raise HabisConfigError.fromException(err, e)

	# Grab the initial guess for reflector positions
	try:
		guessfile = config.get(tsection, 'guessfile')
	except Exception as e:
		err = 'Configuration must specify guessfile in [%s]' % tsection
		raise HabisConfigError.fromException(err, e)

	# Grab the reflector output file
	try:
		outreflector = config.get(tsection, 'outreflector')
	except Exception as e:
		err = 'Configuration must specify outreflector in [%s]' % tsection
		raise HabisConfigError.fromException(err, e)

	try:
		outelements = config.getlist(tsection, 'outelements',
				failfunc=lambda: [''] * len(timefiles))
	except Exception as e:
		err = 'Invalid specification of optional outelements in [%s]' % tsection
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
		c = config.getfloat(tsection, 'c')
		radius = config.getfloat(tsection, 'radius')
	except Exception as e:
		err = 'Configuration must specify c and radius in [%s]' % tsection
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine whether variable sound speeds are allowed
		varc = config.getboolean(tsection, 'varc', failfunc=lambda: False)
	except Exception as e:
		err = 'Invalid specification of optional varc in [%s]' % tsection
		raise HabisConfigError.fromException(err, e)


	# Pull the element indices for each group file
	elements = [np.loadtxt(efile) for efile in inelements]
	# Pull the arrival times and convert surface times to center times
	times = [np.loadtxt(tfile) for tfile in timefiles]
	# Pull the reflector guess as a 2-D matrix
	guess = cutil.asarray(np.loadtxt(guessfile), 2, False)
	# Ensure that the reflector has a sound-speed guess
	if guess.shape[1] != 4:
		guess = np.concatenate([guess, [[c]] * guess.shape[0]], axis=1)

	# Allocate a multiprocessing pool
	pool = multiprocessing.Pool(processes=nproc)

	# Concatenate times and element lists for reflector trilateration
	ctimes = np.concatenate(times, axis=0)
	celts = np.concatenate(elements, axis=0)

	# Compute the reflector positions in parallel
	# Use async calls to correctly handle keyboard interrupts
	result = pool.map_async(getreflpos,
			((t, celts, g[:-1], radius, g[-1], varc) 
				for t, g in izip(ctimes.T, guess)))
	while True:
		try:
			reflectors = np.array(result.get(5))
			break
		except multiprocessing.TimeoutError:
			pass

	# Save the reflector positions
	np.savetxt(outreflector, reflectors, fmt='%16.8f')

	for ctimes, celts, ofile in izip(times, elements, outelements):
		# Don't do trilateration if the result will not be saved
		if not len(ofile): continue

		# Convert arrival times to uniform sound speed
		ctimes *= (reflectors[:,-1] / c)[np.newaxis,:]

		# Create and save a refined estimate of the reflector locations
		# Strip out the sound-speed guess from the reflector position
		pltri = trilateration.PlaneTrilateration(reflectors[:,:-1], radius, c)
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
