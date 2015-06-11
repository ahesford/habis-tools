'''
Routines for performing acoustic trilateration.
'''

import numpy as np, math
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

			J = [ D[i,j] ],

		where entries take the form

			D[i,j] = (centers[i,j] - pos[j]) / dist(i)

		for a distance

			dist(i) = sqrt(sum(centers[i,j] - pos[j], j)).
		'''
		# Ensure that pos is at least a 1-D row vector
		pos = cutil.asarray(pos, 1)

		# Ensure dimensions are compatible
		nrows, ndim = self.centers.shape
		if len(pos) != ndim:
			raise TypeError('Dimensionality of pos does not match Jacobian shape')

		# Compute the spatial variations for this Jacobian
		dist = np.sqrt(np.sum((self.centers - pos[np.newaxis,:])**2, axis=1))
		jac = (self.centers - pos[np.newaxis,:ndim]) / dist[:,np.newaxis]

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

			F[i] = c * times[i] / 2 -
				sqrt(sum((centers[i] - pos[newaxis,:])**2)).
		'''
		# Ensure the arguments are properly formatted
		pos = cutil.asarray(pos, 1)
		times = cutil.asarray(times, 1)

		nrows, ndim = self.centers.shape

		if len(times) != nrows:
			raise TypeError('Arrival time counts must match known element count')

		if len(pos) != ndim:
			raise TypeError('Dimensionality of pos must be compatible with that of known elements')

		dist = np.sqrt(np.sum((self.centers - pos[np.newaxis,:ndim])**2, axis=1))
		return self.c * times / 2 - dist


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


class MultiPointTrilateration(PointTrilateration):
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
		# Initialize according to super
		sup = super(MultiPointTrilateration, self).__init__(centers, c)


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
		for independent, per-element trilateration problems.
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
		sup = super(MultiPointTrilateration, self)
		cfunc = [sup.cost(p, t) for p, t in zip(pos, times)]

		# Concatenate all contributions for the global cost
		return np.concatenate(cfunc)


	def jacobian(self, pos):
		'''
		Computes the Jacobian matrix of the cost function described in
		the docstring for cost(), as a scipy LinearOperator, to
		facilitate Newton-Raphson iteration for simultaneous acoustic
		trilateration of multiple unknown elements. If the object's
		varc property is True,

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
		sup = super(MultiPointTrilateration, self)
		jacs = [sup.jacobian(p) for p in pos]

		# Build the MVP and its adjoint for a LinearOperator
		def mv(x):
			# Reshape the element coordinates for convenience
			x = np.reshape(x, (nelts, ncols), order='C')
			# Compute the independent trilateration parts
			y = [np.dot(j, xv) for j, xv in zip(jacs, x)]
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
			return x.ravel('C')

		jshape = (nelts * nrows, nelts * ncols)
		jac = LinearOperator(shape=jshape, matvec=mv,
				rmatvec=mvt, dtype=self.centers.dtype)
		return jac


	def newton(self, times, pos=None, maxit=100, tol=1e-6, itargs={}):
		'''
		Use Newton-Raphson iteration to recover the positions of
		unknown element associated with the provided round-trip arrival
		times that correspond to known element positions configured for
		the object instance.

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


class PlaneTrilateration(MultiPointTrilateration):
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
		bjac = super(PlaneTrilateration, self).jacobian(pos)

		# Determine the normal to element plane
		normal = facet.lsqnormal(pos)

		# Build the MVP and its adjoint for a LinearOperator
		def mv(x):
			# Use the per-element Jacobian to compute the first part
			y = bjac.matvec(x)
			# Reshape the element coordinates for convenience
			x = np.reshape(x, (nelts, ncols), order='C')
			# Compute the coplanarity parts
			relx = x - np.mean(x, axis=0)
			return np.concatenate([y, np.dot(relx, normal)])

		def mvt(y):
			ntrilat = nelts * nrows
			# Compute the independent trilateration contribution
			x = bjac.rmatvec(y[:ntrilat]).reshape((nelts, ncols), order='C')
			ytri = np.reshape(y[:ntrilat], (nelts, nrows), order='C')
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
		cfunc = super(PlaneTrilateration, self).cost(pos, times)

		# Add costs to enforce coplanarity
		relpos = pos - np.mean(pos, axis=0)
		normal = facet.lsqnormal(pos)
		# Concatenate both contributions for the global cost
		return np.concatenate([cfunc, np.dot(relpos, normal)])
