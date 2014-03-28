'''
Routines for working with transducer facets.
'''

import numpy as np
from numpy import linalg as la


def normal(f):
	'''
	For an N x 3 array of coordinates of transducer elements on a planar
	facet, in which the i-th row represents the coordinates (x, y, z)
	coordinates (across the columns) of the i-th element, find the outward
	facet normal. Outward, in this case, means pointing to the side of the
	plane opposite the origin.
	'''
	# Ensure f is an array
	try: nelts, ndim = f.shape
	except AttributeError:
		f = np.array(f)
		nelts, ndim = f.shape

	# If there aren't enough points on the facet, pick the midpoint
	if nelts < 3:
		n = np.mean(f, axis=0)
		n /= la.norm(n)
		return n

	# The first reference direction points from one corner to another
	ref = f[-1] - f[0]
	ref /= la.norm(ref)
	# Enumerate local directions of all other elements
	vecs = f[1:-1,:] - f[0][np.newaxis,:]
	lvecs = la.norm(vecs, axis=1)
	vecs /= lvecs[:,np.newaxis]
	# Find the vector most orthogonal to the reference
	v = vecs[np.argmin(np.dot(vecs, ref)),:]
	# Compute the normal and its length
	n = np.cross(v, ref)
	nnrm = la.norm(n)

	# If the cross product is very small, treat the elements as collinear
	if nnrm < 1.0e-6:
		n = np.mean(f, axis=0)
		n /= norm(n)
		return n

	# Otherwise, normalize the vector and pick the right direction
	n /= nnrm
	# Find length of normal segment connecting origin to facet plane
	d = np.dot(f[0], n)
	# If d is positive, the normal already points outward
	return n if (d > 0) else -n
