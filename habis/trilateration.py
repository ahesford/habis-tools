'''
Routines for performing acoustic trilateration.
'''

import numpy as np, math
from itertools import izip
from numpy import fft, linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr, LinearOperator
from scipy.optimize import fmin_bfgs
from pycwp import cutil

from . import facet

class ArrivalTimeFinder(object):
	'''
	Given an (Nt x Nt) map of arrival times for signals broadcast by Nt
	transmitter channels (along the rows) and received by a coincident set
	of Nt receiver channels (along the columns), determine a set of Nt
	round-trip arrival times that optimally (in the least-squares sense)
	predict the entire arrival-time map.

	The map can be a masked array to exclude some measurements from
	consideration.
	'''
	def __init__(self, atmap):
		'''
		Initialize an ArrivalTimeFinder using the (optionally masked)
		map of (Nt x Nt) arrival times of signals. The arrival time
		atmap[i][j] is that observed at receive channel j for a
		transmission from channel i.
		'''
		# Copy the arrival-time map
		self.atmap = atmap


	@property
	def atmap(self): return self._atmap

	@atmap.setter
	def atmap(self, atmap):
		# Ensure the map has the right shape
		try: shape = atmap.shape
		except AttributeError:
			atmap = np.array(atmap)
			shape = atmap.shape
		if len(shape) != 2:
			raise TypeError('Arrival time map must be of rank 2')
		if shape[0] != shape[1]:
			raise TypeError('Arrival time map must be square')

		# Capture a copy to the arrival-time map
		self._atmap = atmap.copy()

	@atmap.deleter
	def atmap(self): del self._atmap


	def bfgs(self, itargs={}):
		'''
		Return, using BFGS (scipy.optimize.fmin_bfgs), the optimum
		arrival times based on the previously provided arrival-time
		map. The dictionary itargs is passed to BFGS.
		'''
		# Create a function that computes the cost functional
		def f(x):
			atest = 0.5 * (x[:,np.newaxis] + x[np.newaxis,:])
			return ((atest - self.atmap)**2).sum()
		times = fmin_bfgs(f, np.diag(self.atmap), **itargs)
		return times


	def lsmr(self, itargs={}):
		'''
		Return, using LSMR (scipy.sparse.linagl.lsmr), the optimum
		arrival times based on the previously provided arrival-time
		map. The dictionary itargs is passed to LSMR.
		'''
		# Build the sparse matrix relating optimum times to the map
		nx, ny = self.atmap.shape
		data, indices, indptr = [], [], [0]
		for i in range(nx):
			for j in range(ny):
				if i == j:
					indices.append(i)
					data.append(1.)
				else:
					indices.extend([min(i, j), max(i,j)])
					data.extend([0.5, 0.5])
				indptr.append(len(data))
		matrix = csr_matrix((data, indices, indptr), dtype=np.float32)

		# Flatten the map
		atmap = self.atmap.ravel()
		# Try to determine a mask of values to keep
		try: mask = np.logical_not(atmap.mask)
		except AttributeError: mask = np.ones(atmap.shape, dtype=bool)

		# Solve the system
		times = lsmr(matrix[mask], atmap[mask], **itargs)[0]
		return times


class MultiPointTrilateration(object):
	'''
	Use the fixed positions of known elements, embedded in a medium of
	known wave speed, to simultaneously determine the unknown positions of
	additional elements using the measured arrival times of signals
	reflected from the known elements.

	Optionally, the sound speed can be recovered along with the positions.
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


	def cost(self, pos, times, c=None):
		'''
		Compute the cost function associated with Newton-Raphson
		iterations for acoustic trilateration as configured in the
		object instance. The rank-2 position array pos specifies,
		row-wise, the estimated coordinates of all unknown elements.

		The rank-2 array times specifies along its rows arrival times
		that characterize the separations between the corresponding
		element in pos and each of the previously configured elements.

		The value c is the wave speed to use when evaluating the cost.
		If c is None, the value of self.c is used instead.

		The cost function is the concatenation of the cost functions
		for independent, per-element trilateration problems. A block k
		of the cost function takes the form

			F[k,i] = c * times[k,i] / 2 -
				sqrt(sum((centers[i] - pos[k])**2)).
		'''
		# Treat the positions and times as a rank-2 array
		pos = cutil.asarray(pos, 2, False)
		times = cutil.asarray(times, 2, False)

		# Make sure the sound speed is appropriate
		if c is None: c = self.c

		# The position vector has one extra row for normal components
		nelts, ncols = pos.shape
		nrows, ndim = self.centers.shape

		if ncols != ndim:
			raise TypeError('Element positions must have same dimensionality as reference points')
		if times.shape[0] != nelts:
			raise TypeError('Numbers of rows in pos must match that of times')
		if times.shape[1] != nrows:
			raise TypeError('Per-element arrival time counts must match known element count')

		# Compute distance matrix (nelts, nrows)
		cenpad = self.centers[np.newaxis,:,:]
		pospad = pos[:,np.newaxis,:]
		dist = np.sqrt(np.sum((cenpad - pospad)**2, axis=-1))

		# Flatten in proper order and include time term
		return c * times.ravel('C') / 2 - dist.ravel('C')


	def jacobian(self, pos, times=None):
		'''
		Computes the Jacobian matrix of the cost function described in
		the docstring for cost(), as a scipy LinearOperator, to
		facilitate Newton-Raphson iteration for simultaneous acoustic
		trilateration of multiple unknown elements.

		Each row of the rank-2 array pos specifies the current position
		estimate for one of the unknown elements.

		The Jacobian consists of a block-diagonal part whose blocks
		correspond to independent trilateration problems to recover
		element positions. A Jacobian block J[k] takes the form

			J[k] = [ D[k,i,j] ],

		where entries take the form

			D[k,i,j] = (centers[i,j] - pos[k,j]) / dist(k,i)

		for a distance

			dist(k,i) = sqrt(sum(centers[i,j] - pos[k,j], j)).

		If times is not None, the Jacobian includes an extra column
		taking the form

			times.ravel('C') / 2

		to model variations in wave speed.

		The vector acted on by the LinearOperator should follow the
		form of the argument pos, flattened in row-major order.
		'''
		# Treat the positions and times as rank-2 arrays
		pos = cutil.asarray(pos, 2, False)
		times = cutil.asarray(times, 2, False)

		nelts, ncols = pos.shape
		nrows, ndim = self.centers.shape

		varc = times is not None

		if ncols != ndim:
			raise TypeError('Element positions must have same dimensionality of reference points')
		if varc:
			if times.shape[0] != nelts:
				raise TypeError('Numbers of rows in pos must match that of times')
			if times.shape[1] != nrows:
				raise TypeError('Number of columns in times must match rows in self.centers')

		# Compute distance matrix (nelts, nrows)
		cenpad = self.centers[np.newaxis,:,:]
		pospad = pos[:,np.newaxis,:]
		dist = np.sqrt(np.sum((cenpad - pospad)**2, axis=-1))
		# Compute per-element spatial Jacobians in (nelts, nrows, ndim) matrix
		jacs = (cenpad - pospad) / dist[:,:,np.newaxis]

		# Figure the shape of the Jacobian
		jshape = (nelts * nrows, nelts * ndim + (1 if varc else 0))

		# Build the block-diagonal spatial MVP and adjoint
		def mv(x):
			# Treat element coordinates blockwise
			x = np.reshape(x, (nelts, ndim), order='C')
			# Fill output vector blockwise
			y = np.empty((nelts, nrows), dtype=self.centers.dtype)
			for yv, j, xv in izip(y, jacs, x):
				yv[:] = np.dot(j, xv)
			return y.ravel('C')

		def mvt(y):
			# Treat input vector blockwise
			y = np.reshape(y, (nelts, nrows), order='C')
			# Fill output vector blockwise
			x = np.empty((nelts, ndim), dtype=self.centers.dtype)
			for xv, j, yv in izip(x, jacs, y):
				xv[:] = np.dot(j.T, yv)
			return x.ravel('C')

		if varc:
			# Add a component for speed variations
			smv = mv
			def mv(x):
				# Add the speed contribution to the spatial part
				return smv(x[:-1]) + x[-1] * times.ravel('C')

			smvt = mvt
			def mvt(y):
				# Make an output array with room for the speed part
				xe = np.empty(jshape[1], dtype=self.centers.dtype)
				#  Fill in the spatial part
				xe[:-1] = smvt(y)
				# Compute the speed part
				xe[-1] = np.dot(times.ravel('C'), y)
				return xe

		jac = LinearOperator(shape=jshape, matvec=mv,
				rmatvec=mvt, dtype=self.centers.dtype)
		return jac


	def newton(self, times, pos=None, varc=False, maxit=100, tol=1e-6, itargs={}):
		'''
		Use Newton-Raphson iteration to recover the positions of
		unknown element associated with the provided round-trip arrival
		times that correspond to known element positions configured for
		the object instance.

		Initial estimates pos may be specified, but are assumed to be
		zero by default. The rank-2 array pos should specifies the
		estimated coordinates of one element along each row.

		If varc is True, the wave speed is treated as a variable and
		will be recovered along with the element positions. The initial
		guess for wave speed is always self.c.

		Iterations will stop after maxit iterations or when the norm of
		a computed update is less than tol times the norm of the guess
		used to produce the update, whichever occurs first.

		The inverse of the Jacobian is computed using the LSMR
		algorithm (scipy.sparse.linalg.lsmr). The dictionary itargs is
		passed to the LSMR function as its kwargs.

		The return value is an array of shape (nelts, ndim), where
		nelts is the number of rows in times and ndim is the
		dimensionality of each element in self.centers. If varc is
		True, a second return value will be the recovered sound speed.
		'''
		times = cutil.asarray(times, 2, False)
		pos = cutil.asarray(pos, 2, False)

		nrows, ndim = self.centers.shape
		nelts = times.shape[0]

		if times.shape[1] != nrows:
			raise TypeError('Per-element arrival time counts must match known element count')

		# Copy a provided guess to avoid corrupting input
		if pos is not None: pos = np.array(pos)
		else: pos = np.zeros((nelts, ndim), dtype=self.centers.dtype)

		if pos.shape != (nelts, ndim):
			raise TypeError('Shape of pos must be (times.shape[0], self.centers.shape[1])')

		c = self.c
		n = nelts * ndim

		for i in range(maxit):
			# Build the Jacobian and right-hand side
			jac = self.jacobian(pos, times if varc else None)
			cost = self.cost(pos, times, c)
			# Use LSMR to invert the system
			delt = lsmr(jac, cost, **itargs)[0]
			# Check for convergence
			conv = (la.norm(delt[:n]) < tol * la.norm(pos))
			# Add the update and break when converged
			pos -= delt[:n].reshape((nelts, ndim), order='C')
			if varc: c -= delt[-1]
			if conv: break

		return (pos, c) if varc else pos


class PlaneTrilateration(MultiPointTrilateration):
	'''
	Use the fixed positions of known elements, embedded in a medium of
	known wave speed, to determine the unknown position of a collection of
	additional elements using the measured arrival times of signals
	reflected from the known elements. As additional constraints, the
	positions of the recovered elements will be constrained to
	approximately lie on a plane.
	'''
	def jacobian(self, pos, *args, **kwargs):
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
		estimate for one of the unknown elements. Extra args and kwargs
		are passed to super.jacobian() along with pos.

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
		bjac = super(PlaneTrilateration, self).jacobian(pos, *args, **kwargs)
		bjshape = bjac.shape

		# Figure the shape of the modified Jacobian
		jshape = (bjshape[0] + nelts, bjshape[1])

		# Determine the normal to element plane
		normal = facet.lsqnormal(pos)

		# Build the MVP and its adjoint for a LinearOperator
		def mv(x):
			y = np.empty((jshape[0],), dtype=self.centers.dtype)
			# Compute the per-element MVP
			y[:bjshape[0]] = bjac.matvec(x)
			# Reshape the element coordinates for convenience
			x = np.reshape(x[:nelts*ndim], (nelts, ndim), order='C')
			# Compute the coplanarity parts
			relx = x - np.mean(x, axis=0)
			y[bjshape[0]:] = np.dot(relx, normal)
			return y

		def mvt(y):
			ntrilat = nelts * nrows
			# Compute the independent trilateration contribution
			x = bjac.rmatvec(y[:bjshape[0]])
			# Include coplanarity contribution
			yplan = y[bjshape[0]:,np.newaxis]
			rely = yplan - np.mean(yplan)
			x[:nelts*ndim] += np.dot(rely, normal[np.newaxis,:]).ravel('C')
			return x

		jac = LinearOperator(shape=jshape, matvec=mv,
				rmatvec=mvt, dtype=self.centers.dtype)
		return jac


	def cost(self, pos, *args, **kwargs):
		'''
		Compute the cost function associated with Newton-Raphson
		iterations for acoustic trilateration as configured in the
		object instance. The rank-2 position array pos specifies,
		row-wise, the estimated coordinates of all unknown elements.

		The cost function is the concatenation of the cost functions
		for independent, per-element trilateration problems, followed
		by constraints that all unknown elements should be coplanar.
		The extra args and kwargs are forwarded to super.cost().
		'''
		# Treat the positions as a rank-2 array
		pos = cutil.asarray(pos, 2, False)

		# The position vector has one extra row for normal components
		nelts, ncols = pos.shape
		nrows, ndim = self.centers.shape

		if ncols != ndim:
			raise TypeError('Element positions must have same dimensionality as reference points')

		n = nelts * nrows

		# Build the per-element trilateration costs
		cfunc = np.empty((n + nelts,), dtype=self.centers.dtype)
		sup = super(PlaneTrilateration, self)
		cfunc[:n] = sup.cost(pos, *args, **kwargs)

		# Add costs to enforce coplanarity
		relpos = pos - np.mean(pos, axis=0)
		normal = facet.lsqnormal(pos)
		# Concatenate both contributions for the global cost
		cfunc[n:] = np.dot(relpos, normal)
		return cfunc
