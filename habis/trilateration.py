'''
Routines for performing acoustic trilateration.
'''

import numpy as np, math
from numpy import fft, linalg as la
from scipy.sparse.linalg import lsmr, LinearOperator
from pycwp import cutil

from . import facet

class PointTrilateration(object):
	'''
	Use the fixed positions of known elements, embedded in a medium of
	known wave speed, to determine the unknown position of an additional
	element using the measured arrival times of signals reflected from the
	known elements.
	'''
	def __init__(self, centers, c=1.507):
		'''
		Create a trilateration object using a collection of elements
		whose positions are specified along rows of the rank-2 array
		centers. The wave speed c is used to relate round-trip arrival
		times to distances.
		'''
		# Make a copy of the centers array, ensure it is rank 2
		self.centers = np.array(centers)
		if np.ndim(centers) != 2:
			raise TypeError('Element centers must be a rank-2 array')
		# Copy the wave speed
		self.c = c


	def jacobian(self, pos):
		'''
		Computes the Jacobian matrix of the cost function described in
		the docstring for cost() to facilitate Newton-Raphson iteration
		for acoustic trilateration of an unknown element relative to
		previously configured known elements.

		The current estimate of the unknown element position are
		provided in the rank-1 array pos. The dimensionality of pos
		should match the dimensionality of the known element
		coordinates.

		The Jacobian takes the form

			J = 2 * [ D[i,j] ],

		where entries take the form

			D[i,j] = centers[i,j] - pos[j].
		'''
		# Ensure that pos is at least a 2-D row vector
		pos = cutil.asarray(pos, 1)

		# Ensure dimensions are compatible
		nrows, ndim = self.centers.shape
		if len(pos) != ndim:
			raise TypeError('Dimensionality of pos does not match Jacobian shape')

		# Compute the spatial variations for this Jacobian
		jac = 2 * (self.centers - pos[np.newaxis,:ndim])

		return jac


	def cost(self, pos, times):
		'''
		Computes the cost function associated with Newton-Raphson
		iterations for acoustic trilateration as configured in the
		object instance. The estimated position of an unknown element
		is specified in the rank-1 array pos. Round-trip arrival times
		that characterize the distance between the unknown element and
		all known elements are specified in the rank-1 array times.

		The cost function takes the form

			F[i] = c**2 * (times[i] / 2)**2 -
				sum((centers[i] - pos[newaxis,:])**2).
		'''
		# Ensure the arguments are properly formatted
		pos = cutil.asarray(pos, 1)
		times = cutil.asarray(times, 1)

		nrows, ndim = self.centers.shape

		if len(times) != nrows:
			raise TypeError('Arrival time counts must match known element count')

		# Account for a one-way signal delay, if provided
		if len(pos) != ndim:
			raise TypeError('Dimensionality of pos must be compatible with that of known elements')

		atimes = times / 2.
		dist = np.sum((self.centers - pos[np.newaxis,:ndim])**2, axis=1)
		return (self.c * atimes)**2 - dist


	def newton(self, times, pos=None, maxit=100, tol=1e-6, itargs={}):
		'''
		Use Newton-Raphson iteration to recover the position of an
		unknown element associated with the provided round-trip arrival
		times that correspond to known element positions configured for
		the object instance.

		An initial estimate pos may be specified, but is assumed to be
		the origin by default. Iterations will stop after maxit
		iterations or when the norm of a computed update is less than
		tol times the norm of the guess used to produce the update,
		whichever occurs first.

		The inverse of the Jacobian is computed using the LSMR
		algorithm (scipy.sparse.linalg.lsmr). The dictionary itargs is
		passed to the LSMR function as its kwargs.
		'''
		# Format the times as a rank-1 array
		times = cutil.asarray(times, 1)
		nrows, ndim = self.centers.shape
		# Ensure that a copy is made if a position guess was specified
		if pos is not None: pos = np.array(pos)
		else: pos = np.zeros((ndim,), dtype=self.centers.dtype)

		for i in range(maxit):
			# Build the Jacobian and right-hand side
			jac = self.jacobian(pos[:ndim])
			cost = self.cost(pos, times)
			# Use LSMR to invert the system
			delt = lsmr(jac, cost, **itargs)[0]
			# Check for convergence
			conv = (la.norm(delt) < tol * la.norm(pos))
			pos -= delt
			if conv: break

		return pos


class PlaneTrilateration(PointTrilateration):
	'''
	Use the fixed positions of known elements, embedded in a medium of
	known wave speed, to determine the unknown position of a collection of
	additional elements using the measured arrival times of signals
	reflected from the known elements. As additional constraints, the
	positions of the recovered elements will be constrained to
	approximately lie on a plane.
	'''
	def jacobian(self, pos):
		'''
		Computes the Jacobian matrix of the cost function described in
		the docstring for cost(), as a scipy LinearOperator, to
		facilitate Newton-Raphson iteration for simultaneous acoustic
		trilateration of multiple unknown elements. As additional
		constraints, all unknown elements are approximately coplanar.
		The plane in which the points are assumed to lie is defined in
		the least-squares sense from the estimate pos. If there is no
		detectable plane (e.g., if the position estimates for all
		points is zero), this is equivalent to eliminating the planar
		constraints.

		Each row of the rank-2 array pos specifies the current position
		estimate for one of the unknown elements.

		The Jacobian consists of a block-diagonal part whose blocks
		correspond to independent trilateration problems to recover
		element positions, followed by rows to enforce the coplanarity
		of all elements.

		The vector acted on by the LinearOperator should follow the
		form of the argument pos, flattened in row-major order.
		'''
		# Treat the positions as a rank-2 array
		pos = cutil.asarray(pos, 2, False)

		nelts, ncols = pos.shape
		nrows, ndim = self.centers.shape

		if ncols != ndim:
			raise TypeError('Element positions must have same dimensionality of reference points')

		# Build the per-element Jacobian blocks
		sup = super(PlaneTrilateration, self)
		jacs = [sup.jacobian(p) for p in pos]

		# Determine the normal to element plane
		normal = facet.lsqnormal(pos)

		# Build the MVP and its adjoint for a LinearOperator
		def mv(x):
			# Reshape the element coordinates for convenience
			x = np.reshape(x, (nelts, ncols), order='C')
			# Compute the independent trilateration parts
			y = [np.dot(j, xv) for j, xv in zip(jacs, x)]
			# Compute the coplanarity parts
			relx = x - np.mean(x, axis=0)
			y.append(np.dot(relx, normal))
			return np.concatenate(y)

		def mvt(y):
			ntrilat = nelts * nrows
			# Reshape the independent trilateration parts
			ytri = np.reshape(y[:ntrilat], (nelts, nrows), order='C')
			# Store the output
			x = np.empty((nelts, ncols), dtype=self.centers.dtype)
			# Compute spatial, independent portion of element positions
			for j, yt, xv in zip(jacs, ytri, x):
				xv[:] = np.dot(j.T, yt)
			# Include coplanarity contribution
			yplan = y[ntrilat:ntrilat+nelts,np.newaxis]
			x += np.dot(yplan - np.mean(yplan), normal[np.newaxis,:])
			return x.ravel('C')

		jshape = (nelts * (nrows + 1), nelts * ncols)
		jac = LinearOperator(shape=jshape, matvec=mv,
				rmatvec=mvt, dtype=self.centers.dtype)
		return jac


	def cost(self, pos, times):
		'''
		Compute the cost function associated with Newton-Raphson
		iterations for acoustic trilateration as configured in the
		object instance. The rank-2 position array pos specifies,
		row-wise, the estimated coordinates of all unknown elements.

		The rank-2 array times specifies along its rows arrival times
		that characterize the separations between the corresponding
		element in pos and each of the previously configured elements.

		The cost function is the concatenation of the cost functions
		for independent, per-element trilateration problems, followed
		by constraints that all unknown elements should be coplanar.
		'''
		# Treat the positions and times as a rank-2 array
		pos = cutil.asarray(pos, 2, False)
		times = cutil.asarray(times, 2, False)

		# The position vector has one extra row for normal components
		nelts, ncols = pos.shape
		nrows, ndim = self.centers.shape

		if ncols != ndim:
			raise TypeError('Element positions must have same dimensionality as reference points')
		if times.shape[0] != nelts:
			raise TypeError('Numbers of rows in pos must match that of times')
		if times.shape[-1] != nrows:
			raise TypeError('Per-element arrival time counts must match known element count')

		# Build the per-element trilateration costs
		sup = super(PlaneTrilateration, self)
		cfunc = [sup.cost(p, t) for p, t in zip(pos, times)]

		# Add costs to enforce coplanarity
		relpos = pos - np.mean(pos, axis=0)
		normal = facet.lsqnormal(pos)
		cfunc.append(np.dot(relpos, normal))

		# Concatenate all contributions for the global cost
		return np.concatenate(cfunc)


	def newton(self, times, pos=None, maxit=100, tol=1e-6, itargs={}):
		'''
		Use Newton-Raphson iteration to recover the positions of
		unknown element associated with the provided round-trip arrival
		times that correspond to known element positions configured for
		the object instance. The positions are required to be coplanar.

		Initial estimates pos may be specified, but are assumed to be
		zero by default. The rank-2 array pos should specifies the
		estimated coordinates of one element along each row.
		
		Iterations will stop after maxit iterations or when the norm of
		a computed update is less than tol times the norm of the guess
		used to produce the update, whichever occurs first.

		The inverse of the Jacobian is computed using the LSMR
		algorithm (scipy.sparse.linalg.lsmr). The dictionary itargs is
		passed to the LSMR function as its kwargs.
		'''
		times = cutil.asarray(times, 2)

		nrows, ndim = self.centers.shape
		nelts = times.shape[0]

		if times.shape[1] != nrows:
			raise TypeError('Per-element arrival time counts must match known element count')

		if pos is not None: pos = np.array(pos)
		else: pos = np.zeros((nelts, ndim), dtype=self.centers.dtype)

		if pos.shape[1] != ndim:
			raise TypeError('Element positions must specify an arrival delay parameter')
		if pos.shape[0] != nelts:
			raise TypeError('Number of rows in pos must match that of times')

		for i in range(maxit):
			# Build the Jacobian and right-hand side
			jac = self.jacobian(pos)
			cost = self.cost(pos, times)
			# Use LSMR to invert the system
			delt = lsmr(jac, cost, **itargs)[0]
			# Check for convergence
			conv = (la.norm(delt) < tol * la.norm(pos))
			# Add the update and break when converged
			pos -= delt.reshape((nelts, ndim), order='C')
			if conv: break

		return pos
