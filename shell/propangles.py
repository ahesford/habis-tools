#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os, sys, numpy as np
import multiprocessing, queue

from numpy import ma



from pycwp import process

from habis.habiconf import HabisConfigError, HabisConfigParser

def usage(progname):
	print('USAGE: %s <configuration>' % progname, file=sys.stderr)


def wavepaths(elements, reflectors, nargs={}):
	'''
	Given a list of elements grouped by facet, and a collection of
	reflectors, return a tuple (distances, angles) that indicates the
	distances from each element to each reflector and the angle between the
	propagation direction and the element's directivity axis (the normal to
	the facet). The normal is found using habis.facet.lsqnormal, and the
	optional nargs is a dictionary of kwargs to pass after the facet
	element coordinates argument.

	The argument elements should be a sequence of ndarrays such that
	elements[i] is an N[i]-by-3 array of (x, y, z) coordinates for each of
	N[i] elements in the i-th facet, and reflectors should be an Nr-by-3
	array of (x, y, z) center coordinates for each of Nr reflectors.

	The outputs will be lists of ndarrays such that distances[i] and
	angles[i] are N[i]-by-Nr maps of propagation distances or angles,
	respectively, such that entry (j, k) is the measure from element j to
	reflector k. These lists are suitable for passing to np.concatenate(),
	with axis=0, to produce composite element-to-reflector maps.
	'''
	from numpy.linalg import norm
	from habis.facet import lsqnormal
	# Compute the propagation vectors and normalize
	directions = [reflectors[np.newaxis,:,:] - el[:,np.newaxis,:] for el in elements]
	distances = [norm(dirs, axis=-1) for dirs in directions]
	directions = [dirs / dists[:,:,np.newaxis] 
			for dirs, dists in zip(directions, distances)]

	# The normal should point inward, but points outward by default
	normals = [-lsqnormal(el, **nargs) for el in elements]

	# Figure the propagation angles
	thetas = [np.arccos(np.dot(dirs, ne))
			for dirs, ne in zip(directions, normals)]

	return distances, thetas


def propanglesEngine(config):
	'''
	Calculate the propagation angles and distances between elements and a
	set of reflectors.
	'''
	psec = 'propangles'
	msec = 'measurement'

	try:
		# Grab the reflector positions
		rflfile = config.get(psec, 'reflectors')
	except Exception as e:
		err = 'Configuration must specify reflectors in [%s]' % psec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the element positions by facet
		eltfiles = config.getlist(psec, 'elements')
		if len(eltfiles) < 1:
			err = 'Key elements must contain at least one entry'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify elements in [%s]' % psec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine whether to use surface or center distances
		useradius = not config.get(psec, 'centers', mapper=bool, default=True)
	except Exception as e:
		err = 'Invalid specification of optional centers in [%s]' % psec
		raise HabisConfigError.fromException(err, e)

	if useradius: 
		# Grab the reflector radius if necessary
		try:
			radius = config.get(msec, 'radius', mapper=float)
		except Exception as e:
			err = 'Configuration must specify radius in [%s]' % msec
			raise HabisConfigError.fromException(err, e)

	try:
		# Grab the output distance file
		distfile = config.get(psec, 'distances')
	except Exception as e:
		err = 'Configuration must specify distances in [%s]' % psec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the output angle file
		angfile = config.get(psec, 'angles')
	except Exception as e:
		err = 'Configuration must specify angles in [%s]' % psec
		raise HabisConfigError.fromException(err, e)


	# Load the element and reflector positions, then compute distances and angles
	elements = [np.loadtxt(efile) for efile in eltfiles]
	nedim = elements[0].shape[1]
	for el in elements[1:]:
		if el.shape[1] != nedim:
			raise ValueError('Dimensionality of all element files must agree')
	# Ignore an optional sound-speed column in the reflector coordinates
	reflectors = np.loadtxt(rflfile)[:,:nedim]
	distances, thetas = wavepaths(elements, reflectors)

	# Concatenate the distance and angle lists
	distances = np.concatenate(distances, axis=0)
	thetas = np.concatenate(thetas, axis=0)

	# Offset distances by radius, if desired
	if useradius: distances -= radius

	# Save the outputs
	np.savetxt(distfile, distances, fmt='%16.8f')
	np.savetxt(angfile, thetas, fmt='%16.8f')


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Read the configuration file
	try:
		config = HabisConfigParser(sys.argv[1])
	except:
		print('ERROR: could not load configuration file %s' % sys.argv[1], file=sys.stderr)
		usage(sys.argv[0])
		sys.exit(1)

	# Call the calculation engine
	propanglesEngine(config)
