'''
Routines for working with transducer facets.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np
from numpy import linalg as la
from scipy.sparse.linalg import lsmr

from pycwp import cutil


def normal(f):
	'''
	For an N x 3 array of coordinates of transducer elements on a planar
	facet, in which the i-th row represents the coordinates (x, y, z)
	of the i-th element, find the outward facet normal. Outward, in this
	case, means pointing to the side of the plane opposite the origin.
	'''
	# Ensure f is a rank-2 array (or, at least, a row vector)
	f = cutil.asarray(f, 2, False)
	nelts, ndim = f.shape

	try: eps = np.finfo(f.dtype).eps
	except ValueError: eps = 1.0

	# If there aren't enough points on the facet, pick the midpoint
	if nelts < 3:
		return cutil.vecnormalize(np.mean(f, axis=0))

	# The first reference direction points from one corner to another
	ref = cutil.vecnormalize(f[-1] - f[0])
	# Enumerate local directions of all other elements
	vecs = cutil.vecnormalize(f[1:-1,:] - f[0][np.newaxis,:], axis=1)
	# Find the vector most orthogonal to the reference
	v = vecs[np.argmin(np.dot(vecs, ref)),:]
	# Compute the normal
	n = cutil.vecnormalize(np.cross(v, ref))

	# If the cross product is very small, treat the elements as collinear
	if la.norm(n) < eps:
		return cutil.vecnormalize(np.mean(f, axis=0))

	# Find length of normal segment connecting origin to facet plane
	d = np.dot(f[0], n)
	# If d is positive, the normal already points outward
	return n if (d > 0) else -n


def lsqnormal(f, maxit=100, tol=1e-6, itargs={}):
	'''
	For an N x 3 array of coordinates of transducer elements on an
	approximately planar facet, in which the i-th row represents the
	coordinates (x, y, z) of the i-th element, find the outward facet
	normal in the least-squares sense. Outward, in this case, means
	pointing to the side of the plane opposite the origin.

	The least-squares solution is found using Newton-Raphson iteration with
	using a maximum of maxit iterations and a tolerance of tol. The
	dictionary itargs is passed to LSMR to invert the Jacobian.
	'''
	# Ensure f is a rnak-2 array (or, at least, a row vector)
	f = cutil.asarray(f, 2, False)
	nelts, ndim = f.shape

	# Find the midpoint of the facet
	mp = np.mean(f, axis=0)
	# The guess is the vector pointing toward the midpoint
	guess = cutil.vecnormalize(normal(f))

	# If there aren't enough points on the facet, use the midpoint
	if nelts < 3: return guess

	# Compute the separation between each point and the midpoint
	df = f - mp[np.newaxis,:]

	for i in range(maxit):
		# Build the cost function and Jacobian
		cost = np.concatenate([np.dot(df, guess), [np.dot(guess, guess) - 1.]])
		jac = np.concatenate([df, 2 * guess[np.newaxis,:]])
		# Invert, check for convergence and update
		delt = lsmr(jac, cost, **itargs)[0]
		conv = (la.norm(delt) < tol * la.norm(guess))
		guess -= delt
		if conv: break

	# Normalize and check directionality of the normal
	guess = cutil.vecnormalize(guess)
	d = np.dot(mp, guess)
	return guess if (d > 0) else -guess
