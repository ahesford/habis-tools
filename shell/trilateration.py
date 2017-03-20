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
from habis.formats import savetxt_keymat, loadmatlist

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def getreflpos(args):
	'''
	For args = (times, elocs, guess, rad, c, optc, optr, tol), with a list
	of N round-trip arrival times and an N-by-3 matrix elocs of reference
	element locatations, use habis.trilateration.MultiPointTrilateration to
	determine, starting with the provided 3-dimensional guess position, the
	position of a target with radius rad in a medium with wave speed c.

	If optc is True, the MultiPointTrilateration is allowed to optimize for
	wave speed in addition to target position. If optr is True, the
	MultiPointTrilateration is allowed to optimize for target radius in
	addition to position and (possibly) wave speed.

	The argument tol specifies the convergence tolerance passed to
	MultiPointTrilateration.newton.

	The return value is a 5-dimensional vector in which the first three
	dimensions are the target position, the fourth dimension is the
	optimized wave speed, and the fifth is the optimized target radius. If
	the wave speed or the radius are not optimized, the original value is
	returned in place of an optimized one.
	'''
	times, elemlocs, guess, rad, c, optc, optr, tol = args
	t = trilateration.MultiPointTrilateration(elemlocs, optc, optr)
	rval = t.newton(times, pos=guess, c=c, r=rad, tol=tol)
	# Unpack the recovered solution
	if not (optc or optr):
		pos = rval
	else:
		pos = rval[0]
		if optc: c = rval[1]
		if optr: rad = rval[-1]

	# Expand the position to include wave speed and radius
	nt = pos.shape[0]
	pos = np.concatenate([pos, [[c]] * nt, [[rad]] * nt], axis=1)
	return pos.squeeze()


def geteltpos(args):
	'''
	For args = (elts, eltpos, times, reflectors, rad, c, tol, planewt), where:
	  * elts is a list of element indices,
	  * eltpos maps element indices to guess coordinates (x, y, z),
	  * times maps element indices to a length-N sequence of round-trip
	    arrival times from that element to each of N reflectors,
	  * reflectors is an array of shape (N, 5) that specifies values
	    (x, y, z, c, r) across columns, specifying the position (x, y, z),
	    background wave speed c, and radius r for each of N reflectors,
	  * c is an arbitrary (uniform) background wave speed,
	  * rad is an arbitrary (uniform) reflector radius,
	  * tol is a tolerance for Newton-Raphson iteraton, and
	  * planewt is a weighting on the emphasis of coplanarity constraints

	use habis.trilateration.PlaneTrilaterion (if len(elts) > 2) or
	habis.trilateration.MultiPointTrilateration to recover the positions of
	each element in elts. Per-reflector arrival times (with a possibly
	unique radius and wave speed) will be converted to effective times for
	a common sound speed and target radius before trilateration.

	The return value is a map from element indices to final coordinates.
	'''
	elts, eltpos, times, reflectors, rad, c, tol, planewt = args

	# Pull the element coordinates
	celts = np.array([eltpos[e] for e in elts])
	# Pull the arrival times, convert to common radius and wave speed
	ctimes = ((np.array([times[e] for e in elts]) *
			(reflectors[:,-2] / c)[np.newaxis,:]) +
			(2 * (reflectors[:,-1] - rad) / c)[np.newaxis,:])
	# Strip sound speed and radius from the reflectors
	reflpos = reflectors[:,:-2]
	# No need to enforce coplanarity for one or two elements
	if len(elts) > 2:
		pltri = trilateration.PlaneTrilateration(reflpos, planewt=planewt)
	else: pltri = trilateration.MultiPointTrilateration(reflpos)
	repos = pltri.newton(ctimes, pos=celts, c=c, r=rad, tol=tol)

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
		optc = config.get(tsec, 'optc', mapper=bool, default=False)
	except Exception as e:
		err = 'Invalid specification of optional optc in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine whether variable target radii are allowed
		optr = config.get(tsec, 'optr', mapper=bool, default=False)
	except Exception as e:
		err = 'Invalid specification of optional optr in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	try:
		# Pull the facet (coplanar element groups) size
		fctsize = config.get(tsec, 'fctsize', mapper=int, default=1)
		if fctsize < 1: raise ValueError('fctsize must be a positive integer')
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

	try:
		# Pull the weight for coplanarity constraints
		planewt = config.get(tsec, 'planewt', mapper=float, default=1.0)
		if planewt < 0: raise ValueError('planewt must be nonnegative')
	except Exception as e:
		err = 'Invalid specification of optional planewt in [%s]' % tsec
		raise HabisConfigError.fromException(err, e)

	# Accumulate all element coordinates and arrival times
	eltpos = loadmatlist(inelements)
	times = { k[0]: v
			for k, v in loadmatlist(timefiles, nkeys=2).iteritems()
			if k[0] == k[1] }
	# Only consider elements in both sets
	elements = sorted(set(eltpos).intersection(times))

	# Pull the reflector guess as a 2-D matrix
	guess = np.loadtxt(guessfile, ndmin=2)
	nt, nd = guess.shape
	# Ensure that the reflector has a guess for wave speed and radius
	if nd == 3:
		guess = np.concatenate([guess, [[c, radius]] * nt], axis=1)
	elif nd == 4:
		guess = np.concatenate([guess, [[radius]] * nt], axis=1)
	elif nd != 5:
		raise ValueError('Guess file must contain 3, 4, or 5 columns')

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
				((t, celts, g[:-2], g[-1], g[-2], optc, optr, tol)
					for t, g in izip(ctimes.T, guess)))
		while True:
			try:
				reflectors = np.array(result.get(5))
				break
			except multiprocessing.TimeoutError:
				pass

		# Save the reflector positions
		np.savetxt(outreflector, reflectors, fmt='%16.8f')

		rfldist = norm(reflectors[:,:-2] - guess[:,:-2], axis=-1)
		print 'Iteration', rnd, 'mean reflector shift', np.mean(rfldist), 'stdev', np.std(rfldist)

		# Skip trilateration of element positions if there is no ouptut file
		if not outelements: break

		# Replace guess with reflector positions for next round
		guess = reflectors

		# Compute the element positions in parallel by facet
		# Use async calls to correctly handle keyboard interrupts
		cargs = (eltpos, times, reflectors, radius, c, tol, planewt)
		result = pool.map_async(geteltpos, 
				((elts,) + cargs for elts in facets.itervalues()))
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
		savetxt_keymat(outelements, relements, fmt=refmt)

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
