'''
Routines for performing acoustic trilateration.
'''

import numpy as np, math
from numpy import fft, linalg as la
from scipy.sparse.linalg import lsmr, LinearOperator
from pyajh import cutil

from . import facet

class PointTrilateration(object):
	'''
	Use the fixed positions of known elements, embedded in a medium of
	known wave speed, to determine the unknown position of an additional
	element (and, optionally, a fixed delay associated with that element)
	using the measured arrival times of signals reflected from the known
	elements.
	'''
	def __init__(self, centers, c=1.507):
		'''
		Create a trilateration object using a collection of elements
		whose positions are specified along rows of the rank-2 array
		centers. The wave speed c is used to relate round-trip arrival
		times to distances. If delay is True, the trilateration
		operation will recover an unknown round-trip delay that offsets
		all arrival times.
		'''
		# Make a copy of the centers array, ensure it is rank 2
		self.centers = np.array(centers)
		if np.ndim(centers) != 2:
			raise TypeError('Element centers must be a rank-2 array')
		# Copy the wave speed
		self.c = c


	def jacobian(self, pos, times=None):
		'''
		Computes the Jacobian matrix of the cost function described in
		the docstring for cost() to facilitate Newton-Raphson iteration
		for acoustic trilateration of an unknown element relative to
		previously configured known elements.

		The current estimate of the unknown element position are
		provided in the rank-1 array pos. When times is None, the
		dimensionality in pos should match the dimensionality of the
		known element coordinates. Otherwise, pos should contain an
		extra dimension to specify the current estimate of a one-way
		arrival delay, tau, associated with the unknown element.

		The rank-1 array times, when provided, specifies round-trip
		arrival times that characterize the separation between the
		unknown element and each of the known elements. The Jacobian
		produced when times is not None will contain an extra column
		corresponding to variations in the one-way arrival delay tau.

		The Jacobian takes the form

			J = 2 * [ D[i,j] | T[i] ],

		where T[i] is a column vector with elements

			T[i] = c**2 * (tau - times[i] / 2)

		and is only included when times is not None. The remaining
		entries take the form

			D[i,j] = centers[i,j] - pos[j].
		'''
		# Ensure that pos and times are at least 2-D row vectors
		pos = cutil.asarray(pos, 1)
		times = cutil.asarray(times, 1)

		# Ensure dimensions are compatible
		nrows, ndim = self.centers.shape
		ncols = ndim + int(times is not None)
		if len(pos) != ncols:
			raise TypeError('Dimensionality of pos does not match Jacobian shape')
		if times is not None:
			# Check the shape of the times array
			if len(times) != nrows:
				raise TypeError('Arrival time counts must match known element count')

		# Compute the spatial variations for this Jacobian
		jac = np.empty((nrows, ncols), dtype=self.centers.dtype)
		jac[:,:ndim] = 2 * (self.centers - pos[np.newaxis,:ndim])
		# Include delay contributions if necessary
		if times is not None:
			jac[:,ndim] = 2 * self.c**2 * (pos[-1] - times / 2.)

		return jac


	def cost(self, pos, times):
		'''
		Computes the cost function associated with Newton-Raphson
		iterations for acoustic trilateration as configured in the
		object instance. The estimated position of an unknown element
		is specified in the rank-1 array pos. Round-trip arrival times
		that characterize the distance between the unknown element and
		all known elements are specified in the rank-1 array times. 

		If pos contains one more dimension than the known element
		positions, the last dimension represents an estimate of a
		one-way arrival delay. Otherwise, no delay is assumed to exist.

		The cost function takes the form

			F[i] = c**2 * (times[i] / 2 - pos[-1])**2 
			       - sum((centers[i] - pos[newaxis,:])**2),

		where, in this example, pos is always assumed to include the
		unknown delay.
		'''
		# Ensure the arguments are properly formatted
		pos = cutil.asarray(pos, 1)
		times = cutil.asarray(times, 1)

		nrows, ndim = self.centers.shape

		if len(times) != nrows:
			raise TypeError('Arrival time counts must match known element count')

		# Account for a one-way signal delay, if provided
		tau = 0.
		if len(pos) == ndim + 1: tau = pos[-1]
		elif len(pos) != ndim:
			raise TypeError('Dimensionality of pos must be compatible with that of known elements')

		atimes = times / 2. - tau
		dist = np.sum((self.centers - pos[np.newaxis,:ndim])**2, axis=1)
		return (self.c * atimes)**2 - dist


	def newton(self, times, pos=None, usedelay=False, maxit=100, tol=1e-6, itargs={}):
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

		If usedelay is True, an unknown one-way delay is modeled and
		determined from the iteration. Otherwise, the delay is assumed
		to be zero unless the initial position estimate incorporates a
		fixed delay as its last coordinate.

		The inverse of the Jacobian is computed using the LSMR
		algorithm (scipy.sparse.linalg.lsmr). The dictionary itargs is
		passed to the LSMR function as its kwargs.
		'''
		# Format the times as a rank-1 array
		times = cutil.asarray(times, 1)
		nrows, ndim = self.centers.shape
		ncols = ndim + int(usedelay)
		# Ensure that a copy is made if a position guess was specified
		if pos is not None: np.array(pos)
		else: pos = np.zeros((ncols,), dtype=self.centers.dtype)

		for i in range(maxit):
			# Build the Jacobian and right-hand side
			if usedelay: jac = self.jacobian(pos, times)
			else: jac = self.jacobian(pos[:ndim], None)
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
	additional elements along with a per-element, one-way arrival delay
	using the measured arrival times of signals reflected from the known
	elements. As additional constraints, the positions of the recovered
	elements will be constrained to approximately lie on a plane. The
	average element delay is constrained to be approximately zero, so the
	arrival times should have a real average delay subtracted before
	trilateration is attempted.
	'''
	def jacobian(self, pos, times):
		'''
		Computes the Jacobian matrix of the cost function described in
		the docstring for cost(), as a scipy LinearOperator, to
		facilitate Newton-Raphson iteration for simultaneous acoustic
		trilateration of multiple unknown elements. As additional
		constraints, all unknown elements are approximately coplanar
		and have a mean one-way delay of zero. The plane in which the
		points are assumed to lie is defined in the least-squares sense
		from the estimate pos. If there is no detectable plane (e.g.,
		if the position estimates for all points is zero), this is
		equivalent to eliminating the planar constraints.

		Each row of the rank-2 array pos specifies the current position
		estimate for one of the unknown elements. The positions should
		contain an extra dimension (column) to specify the current
		estimates of per-element, one-way arrival delays.

		The rank-2 array times specifies along its rows the arrival
		times that characterize separations between the corresponding
		unknown element in pos and each of the previously configured,
		known elements. 
		
		The Jacobian consists of a block-diagonal part whose blocks
		correspond to independent trilateration problems to recover
		element positions and unknown delays, followed by rows to
		enforce the coplanarity of all elements, and one row to enforce
		the zero mean of the delays. The Jacobian also has additional
		columns corresponding to variations in the normal direction of
		the containing plane.

		The vector acted on by the LinearOperator should follow the
		form of the argument pos, flattened in row-major order.
		'''
		# Treat the positions and times as a rank-2 array
		pos = cutil.asarray(pos, 2, False)
		times = cutil.asarray(times, 2, False)

		if times is None:
			raise TypeError('Arrival times are not optional for planar trilateration')

		nelts, ncols = pos.shape
		nrows, ndim = self.centers.shape

		if ncols != ndim + 1:
			raise TypeError('Element positions must specify an arrival delay parameter')
		if times.shape[-1] != nrows:
			raise TypeError('Per-element arrival time counts must match known element count')
		if times.shape[0] != nelts:
			raise TypeError('Numbers of rows in pos must match that of times')

		# Build the per-element Jacobian blocks
		sup = super(PlaneTrilateration, self)
		jacs = [sup.jacobian(p, t) for p, t in zip(pos, times)]

		# Determine the normal to the plane of the elements
		normal = facet.lsqnormal(pos[:,:ndim])

		# Build the MVP and its adjoint for a LinearOperator
		def mv(x):
			# Reshape the element coordinates for convenience
			x = np.reshape(x, (nelts, ncols), order='C')
			# Compute the independent trilateration parts
			y = [np.dot(j, xv) for j, xv in zip(jacs, x)]
			# Compute the coplanarity parts
			relx = x[:,:ndim] - np.mean(x[:,:ndim], axis=0)
			y.append(np.dot(relx, normal))
			# Compute the mean delay part
			y.append([np.sum(x[:,-1])])
			return np.concatenate(y)

		def mvt(y):
			ntrilat = nelts * nrows
			# Reshape the independent trilateration parts
			ytri = np.reshape(y[:ntrilat], (nelts, nrows), order='C')
			# Pull the coplanar parts and compute a mean
			yplan = y[ntrilat:ntrilat+nelts]
			ypmean = np.mean(yplan)
			# Store the output
			x = np.empty((nelts, ncols), dtype=self.centers.dtype)
			# Compute the spatial portion of the element positions
			for j, yt, yp, xv in zip(jacs, ytri, yplan, x):
				# Compute the independent trilateration part
				xv[:] = np.dot(j.T, yt)
				# Include coplanarity contribution
				# (Does not affect optional delays)
				xv[:ndim] += normal * (yp - ypmean)
			# Include mean-delay contribution
			x[:,-1] += y[-1]
			return x.ravel('C')

		jshape = (nelts * (nrows + 1) + 1, nelts * ncols)
		jac = LinearOperator(shape=jshape, matvec=mv,
				rmatvec=mvt, dtype=self.centers.dtype)
		return jac


	def cost(self, pos, times):
		'''
		Compute the cost function associated with Newton-Raphson
		iterations for acoustic trilateration as configured in the
		object instance. The rank-2 position array pos specifies,
		row-wise, the estimated coordinates of all unknown elements.
		The positions must include an extra dimension (column) which
		specifies a per-element, one-way arrival delay.

		The rank-2 array times specifies along its rows arrival times
		that characterize the separations between the corresponding
		element in pos and each of the previously configured elements.

		The cost function is the concatenation of the cost functions
		for independent, per-element trilateration problems, followed
		by constraints that all unknown elements should be coplanar and
		that the mean delay should be zero.
		'''
		# Treat the positions and times as a rank-2 array
		pos = cutil.asarray(pos, 2, False)
		times = cutil.asarray(times, 2, False)

		# The position vector has one extra row for normal components
		nelts, ncols = pos.shape
		nrows, ndim = self.centers.shape

		if ncols != ndim + 1:
			raise TypeError('Element positions must specify an arrival delay parameter')
		if times.shape[0] != nelts:
			raise TypeError('Numbers of rows in pos must match that of times')
		if times.shape[-1] != nrows:
			raise TypeError('Per-element arrival time counts must match known element count')

		# Build the per-element trilateration costs
		sup = super(PlaneTrilateration, self)
		cfunc = [sup.cost(p, t) for p, t in zip(pos, times)]

		# Add costs to enforce coplanarity
		relpos = pos[:,:ndim] - np.mean(pos[:,:ndim], axis=0)
		normal = facet.lsqnormal(pos[:,:ndim])
		cfunc.append(np.dot(relpos, normal))

		# Add a cost to enfoce the zero-mean delays
		cfunc.append([np.sum(pos[:,-1])])

		# Concatenate all contributions for the global cost
		return np.concatenate(cfunc)


	def newton(self, times, pos=None, usedelay=True, maxit=100, tol=1e-6, itargs={}):
		'''
		Use Newton-Raphson iteration to recover the positions of
		unknown element associated with the provided round-trip arrival
		times that correspond to known element positions configured for
		the object instance. The positions are required to be coplanar.

		Initial estimates pos may be specified, but are assumed to be
		zero by default. The rank-2 array pos should specifies the
		estimated coordinates of one element along each row, with one
		extra column to specify a per-element, one-way arrival delay.
		Additionally, one extra row should be provided that contains
		the estimated components of the unit normal to the plane
		containing all elements. The value in the delay column for this
		row is always ignored.
		
		Iterations will stop after maxit iterations or when the norm of
		a computed update is less than tol times the norm of the guess
		used to produce the update, whichever occurs first.

		The argument usedelay must be True and exists only to maintain
		call signature compatibility with the PointTrilateration
		superclass.

		The inverse of the Jacobian is computed using the LSMR
		algorithm (scipy.sparse.linalg.lsmr). The dictionary itargs is
		passed to the LSMR function as its kwargs.
		'''
		if not usedelay:
			raise ValueError('Parameter usedelay must be True in this subclass')
		times = cutil.asarray(times, 2)

		nrows, ndim = self.centers.shape
		nelts, ncols = times.shape[0], ndim + 1

		if times.shape[1] != nrows:
			raise TypeError('Per-element arrival time counts must match known element count')

		if pos is not None: pos = np.array(pos)
		else: pos = np.zeros((nelts, ncols), dtype=self.centers.dtype)

		if pos.shape[1] != ncols:
			raise TypeError('Element positions must specify an arrival delay parameter')
		if pos.shape[0] != nelts:
			raise TypeError('Number of rows in pos must match that of times')

		for i in range(maxit):
			# Build the Jacobian and right-hand side
			jac = self.jacobian(pos, times)
			cost = self.cost(pos, times)
			# Use LSMR to invert the system
			delt = lsmr(jac, cost, **itargs)[0]
			# Check for convergence
			conv = (la.norm(delt) < tol * la.norm(pos))
			# Add the update and break when converged
			pos -= delt.reshape((nelts, ncols), order='C')
			if conv: break

		return pos
