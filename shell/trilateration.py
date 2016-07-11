#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os, sys, itertools, numpy as np
import multiprocessing

from numpy.linalg import norm

from itertools import izip

from pycwp import process

from habis import trilateration
from habis.habiconf import HabisConfigError, HabisConfigParser, matchfiles
from habis.formats import savekeymat, loadkeymat

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def getreflpos(args):
	'''
	For args = (times, elocs, guess, rad, c, varc, tol), with a list of N
	round-trip arrival times and an N-by-3 matrix elocs of reference
	element locatations, use habis.trilateration.MultiPointTrilateration to
	determine, starting with the provided 3-dimensional guess position, the
	position of a reflector with radius rad in a medium with sound speed c.

	If varc is True, the MultiPointTrilateration is allowed to recover
	variable sound speed in addition to element position.

	The argument tol specifies the convergence tolerance passed to
	MultiPointTrilateration.newton.

	The return value is a 4-dimensional vector in which the first three
	dimensions are the element position and the last dimension is the
	recovered sound speed. If varc is False, the fourth dimension just
	copies the input parameter c.
	'''
	times, elemlocs, guess, rad, c, varc, tol = args
	t = trilateration.MultiPointTrilateration(elemlocs, rad, c)
	rval = t.newton(times, pos=guess, varc=varc, tol=tol)
	# Unpack the recovered solution
	pos, c = rval if varc else (rval, c)
	# Expand the position to include sound speed
	pos = np.concatenate([pos, [[c]] * pos.shape[0]], axis=1)
	return pos.squeeze()


def geteltpos(args):
	'''
	For args = (elts, eltpos, times, reflectors, rad, c, tol), where:
	  * elts is a list of element indices,
	  * eltpos maps element indices to guess coordinates (x, y, z),
	  * times maps element indices to a length-N sequence of round-trip
	    arrival times from that element to each of N reflectors,
	  * reflectors is a length-N sequence (x, y, z, c) specifying the
	    position and background sound speed for each reflector,
	  * rad is the (common) radius of the reflectors,
	  * c is the (uniform) background sound speed,
	  * and tol is a tolerance for Newton-Raphson iteraton,

	use habis.trilateration.PlaneTrilaterion (if len(elts) > 2) or
	habis.trilateration.MultiPointTrilateration to recover the positions of
	each element in elts.

	The return value is a map from element indices to final coordinates.
	'''
	elts, eltpos, times, reflectors, rad, c, tol = args

	# Pull the element coordinates
	celts = np.array([eltpos[e] for e in elts])
	# Pull the arrival times, converted to background speed
	ctimes = (np.array([times[e] for e in elts]) *
			(reflectors[:,-1] / c)[np.newaxis,:])

	# No need to enforce coplanarity for one or two elements
	tcls = (trilateration.PlaneTrilateration if len(elts) > 2
			else trilateration.MultiPointTrilateration)
	pltri = tcls(reflectors[:,:-1], rad, c)
	repos = pltri.newton(ctimes, pos=celts, tol=tol)

	return dict(izip(elts, repos))


def trilaterationEngine(config):
	'''
	Use the MultiPointTrilateration and PlaneTrilateration classes in
	habis.trilateration to determine, iteratively from a set of
	measurements of round-trip arrival times, the unknown positions of
	a set of reflectors followed by estimates of the positions of the
	hemisphere elements.
	'''
	tsec = 'trilateration'
	msec = 'measurement'
	try:
		# Try to grab the input time files
		timefiles = matchfiles(config.getlist(tsec, 'timefile'))
		if len(timefiles) < 1:
			err = 'Key timefile must contain at least one entry'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify timefile in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Try to grab the nominal element files
		inelements = matchfiles(config.getlist(tsec, 'inelements'))
	except Exception as e:
		err = 'Configuration must specify inelements in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	# Grab the initial guess for reflector positions
	try:
		guessfile = config.get(tsec, 'guessfile')
	except Exception as e:
		err = 'Configuration must specify guessfile in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	# Grab the reflector output file
	try:
		outreflector = config.get(tsec, 'outreflector')
	except Exception as e:
		err = 'Configuration must specify outreflector in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	try:
		outelements = config.get(tsec, 'outelements', default=None)
	except Exception as e:
		err = 'Invalid specification of optional outelements in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the number of processes to use (optional)
		nproc = config.get('general', 'nproc', mapper=int,
				failfunc=process.preferred_process_count)
	except Exception as e:
		err = 'Invalid specification of optional nproc in [general]'
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the sound speed and radius
		c = config.get(msec, 'c', mapper=float)
		radius = config.get(msec, 'radius', mapper=float)
	except Exception as e:
		err = 'Configuration must specify c and radius in [%s]' % msec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the convergence tolerance
		tol = config.get(tsec, 'tolerance', mapper=float, default=1e-6)
	except Exception as e:
		err = 'Invalid specification of optional tolerance in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine whether variable sound speeds are allowed
		varc = config.get(tsec, 'varc', mapper=bool, default=False)
	except Exception as e:
		err = 'Invalid specification of optional varc in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Pull the facet (coplanar element groups) size
		fctsize = config.get(tsec, 'fctsize', mapper=int, default=1)
	except Exception as e:
		err = 'Invalid specification of optional fctsize in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Pull the maximum iteration count
		maxiter = config.get(tsec, 'maxiter', mapper=int, default=1)
	except Exception as e:
		err = 'Invalid specification of optional maxiter in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Pull the itereation stop distance
		stopdist = config.get(tsec, 'stopdist', mapper=float, default=0)
	except Exception as e:
		err = 'Invalid specification of optional stopdist in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	if fctsize < 1:
		raise HabisConfigError('Optional fctsize must be a positive integer')

	# Accumulate all element coordinates and arrival times
	eltpos = dict(kp for efile in inelements
			for kp in loadkeymat(efile).iteritems())
	times = dict((k[0], v) for tfile in timefiles
			for k, v in loadkeymat(tfile, nkeys=2).iteritems() if k[0] == k[1])
	# Only consider elements in both sets
	elements = sorted(set(eltpos.iterkeys()).intersection(times.iterkeys()))

	# Pull the reflector guess as a 2-D matrix
	guess = np.loadtxt(guessfile, ndmin=2)
	# Ensure that the reflector has a sound-speed guess
	if guess.shape[1] == 3:
		guess = np.concatenate([guess, [[c]] * guess.shape[0]], axis=1)
	elif guess.shape[1] != 4:
		raise ValueError('Guess file must contain 3 or 4 columns')

	# Allocate a multiprocessing pool
	pool = multiprocessing.Pool(processes=nproc)

	# Pull the relevant times
	ctimes = np.array([times[e] for e in elements])
	if ctimes.ndim == 1:
		ctimes = ctimes[:,np.newaxis]

	# Build a list of the elements in each facet
	facets = { }
	for e in elements:
		f = int(e // fctsize)
		try: facets[f].append(e)
		except KeyError: facets[f] = [e]

	for rnd in range(1, maxiter + 1):
		# Pull the relevant element coordinates
		celts = np.array([eltpos[e] for e in elements])

		# Compute the reflector positions in parallel
		# Use async calls to correctly handle keyboard interrupts
		result = pool.map_async(getreflpos,
				((t, celts, g[:-1], radius, g[-1], varc, tol)
					for t, g in izip(ctimes.T, guess)))
		while True:
			try:
				reflectors = np.array(result.get(5))
				break
			except multiprocessing.TimeoutError:
				pass

		# Save the reflector positions
		np.savetxt(outreflector, reflectors, fmt='%16.8f')

		rfldist = norm(reflectors[:,:-1] - guess[:,:-1], axis=-1)
		print 'Iteration', rnd, 'mean reflector shift', np.mean(rfldist), 'stdev', np.std(rfldist)

		# Skip trilateration of element positions if there is no ouptut file
		if not outelements: break

		guess = reflectors

		# Compute the element positions in parallel by facet
		# Use async calls to correctly handle keyboard interrupts
		result = pool.map_async(geteltpos,
				((elts, eltpos, times, reflectors, radius, c, tol)
					for elts in facets.itervalues()))
		while True:
			try:
				relements = dict(kp for r in result.get(5) for kp in r.iteritems())
				break
			except multiprocessing.TimeoutError:
				pass

		reltdist = [norm(v - eltpos[i]) for i, v in relements.iteritems()]
		print 'Iteration', rnd, 'mean element shift', np.mean(reltdist), 'stdev', np.std(reltdist)

		# Save the element coordinates in the output file
		refmt = ['%d'] + ['%16.8f']*(len(relements.itervalues().next()))
		savekeymat(outelements, relements, fmt=refmt)

		if max(rfldist) < stopdist and max(reltdist) < stopdist:
			print 'Convergence achieved'
			break

		eltpos = relements


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
	trilaterationEngine(config)
