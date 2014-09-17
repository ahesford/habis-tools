'''
Routines for working with transducer facets.
'''

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
	except ValueError: eps = 1.0e-7

	# If there aren't enough points on the facet, pick the midpoint
	if nelts < 3:
		n = np.mean(f, axis=0)
		return n / max(la.norm(n), eps)

	# The first reference direction points from one corner to another
	ref = f[-1] - f[0]
	ref /= max(la.norm(ref), eps)
	# Enumerate local directions of all other elements
	vecs = f[1:-1,:] - f[0][np.newaxis,:]
	lvecs = la.norm(vecs, axis=1)
	vecs /= np.fmax(lvecs[:,np.newaxis], eps)
	# Find the vector most orthogonal to the reference
	v = vecs[np.argmin(np.dot(vecs, ref)),:]
	# Compute the normal and its length
	n = np.cross(v, ref)
	nnrm = la.norm(n)

	# If the cross product is very small, treat the elements as collinear
	if nnrm < eps:
		n = np.mean(f, axis=0)
		return n / max(la.norm(n), eps)

	# Otherwise, normalize the vector and pick the right direction
	n /= nnrm
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

	try: eps = np.finfo(f.dtype).eps
	except ValueError: eps = tol

	# Find the midpoint of the facet
	mp = np.mean(f, axis=0)
	# If the norm is small, don't normalize
	guess = mp / max(la.norm(mp), eps)

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
	guess /= max(la.norm(guess), eps)
	d = np.dot(mp, guess)
	return guess if (d > 0) else -guess
