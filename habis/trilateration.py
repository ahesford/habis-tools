'''
Routines for performing acoustic trilateration.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, math

from numpy import ma, fft, linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr, LinearOperator
from scipy.optimize import fmin_bfgs
from pycwp import cutil

from . import facet

class ArrivalTimeFinder(object):
	'''
	Given an (Nt x Nr) map of arrival times for signals broadcast by Nt
	transmitter channels (along the rows) and received by a set of Nr
	receiver channels (along the columns), determine a set of Nr round-trip
	arrival times that optimally (in the least-squares or minimum norm
	sense) predict the entire arrival-time map.

	The map can be a masked array to exclude some measurements from
	consideration.
	'''
	def __init__(self, *args, **kwargs):
		'''
		Initialize an ArrivalTimeFinder by passing all arguments to
		self.setatmap().
		'''
		# Assign the arrival-time map
		self.setatmap(*args, **kwargs)


	@property
	def atmap(self):
		''' A reference to the associated arrival-time map. '''
		return self._atmap

	@property
	def rxelts(self):
		''' The list of receive elements represented in the arrival-time map. '''
		return self._rxelts

	@property
	def txelts(self):
		''' The list of transmit elements represented in the arrival-time map. '''
		return self._txelts


	def setatmap(self, atmap, txelts=None, rxelts=None):
		'''
		Assign the given arrival-time map, which characterizes the
		arrival time from transmission element txelts[i] to receive
		element rxelts[j] as atmap[i,j].

		The map may be a masked array; masked values will not be used
		in optimization.

		If atmap is an array and txelts is None, range(atmap.shape[0])
		is assumed. Likewise, if atmap is an array and rxelts is None,
		range(atmap.shape[1]) is assumed.

		The atmap can also be a dictionary-like object (with a .keys()
		method) which maps (t,r) pairs to arrival-time values. In this
		case, any txelts and rxelts arguments are ignored and replaced
		with the tuples

			txelts = tuple(sorted(set(k[0] for k in atmap.keys())))
			rxelts = tuple(sorted(set(k[1] for k in atmap.keys())))

		Any missing keys in (txelts X rxelts) will be interpreted as
		masked (undesired) values.
		'''
		try:
			# Pull the elements right from dictionary keys
			txelts, rxelts = [tuple(sorted(set(ks)))
						for ks in zip(*atmap.keys())]
		except AttributeError:
			# Copy an array-like map, preserving any masks
			atmap = ma.array(atmap, copy=True) 

			try:
				ntx, nrx = atmap.shape
			except ValueError:
				raise TypeError('Arrival-time map must be of rank 2') 

			if txelts is None: txelts = tuple(range(ntx))
			else:
				if len(txelts) != ntx:
					raise ValueError('Length of txelts must match row count in atmap')
				txelts = tuple(txelts)

			if rxelts is None: rxelts = tuple(range(nrx))
			else:
				if len(rxelts) != nrx:
					raise ValueError('Length of rxelts must match column count in atmap')
				rxelts = tuple(rxelts)
		else:
			# Convert the dictionary to a masked array
			atarr = ma.empty((len(txelts), len(rxelts)), dtype=np.float32)
			for i, t in enumerate(txelts):
				for j, r in enumerate(rxelts):
					try: atarr[i,j] = atmap[(t,r)]
					except KeyError: atarr[i,j] = ma.masked
			atmap = atarr

		# Copy the maps
		self._txelts = txelts
		self._rxelts = rxelts
		self._atmap = atmap


	def lsmr(self, itargs={}):
		'''
		Compute, using LSMR (scipy.sparse.linalg.lsmr), the optimum
		round-trip arrival times based on the previously provided
		arrival-time map and participating element lists. The
		dictionary itargs is passed to LSMR.

		The return value is a dictionary whose keys are element indices
		and whose values are the associated round-trip arrival times.
		'''
		# Build the sparse matrix relating optimum times to the map
		data, indices, indptr = [], [], [0]

		# Build a master, participating element list
		eltlist = sorted(set(self.txelts).union(self.rxelts))
		# Map the element indices to list order
		eltmap = dict(reversed(x) for x in enumerate(eltlist))

		for i in self.txelts:
			ei = eltmap[i]
			for j in self.rxelts:
				ej = eltmap[j]

				if i == j:
					indices.append(ei)
					data.append(1.)
				else:
					indices.extend((min(ei,ej), max(ei,ej)))
					data.extend((0.5, 0.5))

				indptr.append(len(data))

		matrix = csr_matrix((data, indices, indptr), dtype=np.float32)

		# Flatten the map in C-order to agree with rows of relational matrix
		atmap = self.atmap.ravel()
		# Try to determine a mask of values to keep
		mask = np.logical_not(ma.getmaskarray(atmap))

		# Solve the system and build the map
		times = lsmr(matrix[mask], atmap[mask], **itargs)[0]
		return dict(kp for kp in zip(eltlist, times))


class MultiPointTrilateration(object):
	'''
	Use the fixed positions of known elements to determine the unknown
	positions of a collection of targets using the measured arrival times
	of backscatter signals measured by the known elements from the targets.

	The sound speed and radius can be optionally recovered with positions.
	'''
	def __init__(self, centers, optc=False, optr=False):
		'''
		Create a trilateration object using a collection of known
		elements whose center positions are specified along rows of the
		rank-2 array centers.

		If optc is True, Newton optimizations that recover the unknown
		target positions will also recover an optimum wave speed.

		If optr is True, the Newton optimizations will also recover an
		optimum target radius.
		'''
		# Make a copy of the centers array, ensure it is rank 2
		self.centers = np.array(centers)
		if np.ndim(centers) != 2:
			raise TypeError('Element centers must be a rank-2 array')

		# Record the preferences for optimized wave speed and radius
		self.optc = optc
		self.optr = optr


	def cost(self, pos, times, c=1.5, r=0.0):
		'''
		Compute the cost function associated with Newton-Raphson
		iterations for acoustic trilateration as configured in the
		object instance.

		The rank-2 position array pos specifies, row-wise, the
		estimated coordinates of all unknown targets.

		The rank-2 array times specifies along its rows arrival times
		that characterize the separations between the corresponding
		target in pos and each of the previously configured elements.

		When computing the cost, the wave speed is given by the value c
		and unknown targets are all assumed to have a radius r. The
		cost function is a concatenation of individual, per-traget
		trilateration costs. The block for target k takes the form

			F[k,i] = r + c * times[k,i] / 2 -
					sqrt(sum((centers[i] - pos[k])**2)).
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
		if times.shape[1] != nrows:
			raise TypeError('Per-element arrival time counts must match known element count')

		# Compute distance matrix (nelts, nrows)
		cenpad = self.centers[np.newaxis,:,:]
		pospad = pos[:,np.newaxis,:]
		dist = np.sqrt(np.sum((cenpad - pospad)**2, axis=-1))

		# Flatten in proper order and include time term
		return r + c * times.ravel('C') / 2 - dist.ravel('C')


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

		If self.optc is False, the argument times is ignored. When
		self.optc is True, the Jacobian will include an extra column
		taking the form

			times.ravel('C') / 2

		to model variations in wave speed.

		If self.optr is True, the Jacobian will include an extra column
		of all ones to model variations in target radius.

		The vector acted on by the LinearOperator should follow the
		form of the argument pos, flattened in row-major order. A wave
		speed or target radius will need to be appended to the vector
		when self.optc or self.optr, respectively, are True.
		'''
		# Treat the positions and times as rank-2 arrays
		pos = cutil.asarray(pos, 2, False)
		times = cutil.asarray(times, 2, False)

		nelts, ncols = pos.shape
		nrows, ndim = self.centers.shape

		if ncols != ndim:
			raise TypeError('Element positions must have same dimensionality of reference points')
		if self.optc:
			if times is None:
				raise TypeError('Argument times must not be None when self.optc is True')
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
		nx = nelts * ndim
		jshape = (nelts * nrows, nx + int(self.optc) + int(self.optr))

		# Build the block-diagonal spatial MVP and adjoint
		def mv(x):
			# Treat element coordinates blockwise
			x = np.reshape(x, (nelts, ndim), order='C')
			# Create and fill output vector blockwise
			y = np.empty((nelts, nrows), dtype=self.centers.dtype)
			for yv, j, xv in zip(y, jacs, x):
				yv[:] = np.dot(j, xv)
			return y.ravel('C')

		def mvt(y):
			# Treat input vector blockwise
			y = np.reshape(y, (nelts, nrows), order='C')
			# Create an output if one was not provided
			x = np.empty((nelts, ndim), dtype=self.centers.dtype)
			# Fill output vector blockwise
			for xv, j, yv in zip(x, jacs, y):
				xv[:] = np.dot(j.T, yv)
			return x.ravel('C')

		if self.optc:
			# Modify Jacobian operators to account for variable speed
			smv = mv
			def mv(x):
				# Add contribution of final speed column
				return smv(x[:nx]) + x[nx] * times.ravel('C')

			smvt = mvt
			def mvt(y):
				# Output needs room for speed part
				xe = np.empty(nx + 1, dtype=self.centers.dtype)
				# Fill in the spatial part
				xe[:nx] = smvt(y)
				# Fill in the speed part
				xe[nx] = np.dot(times.ravel('C'), y)
				return xe

		if self.optr:
			# Modify Jacobian operators to account for variable radius
			pmv = mv
			def mv(x):
				# A new last column accounts for variable radius
				return pmv(x[:-1]) + x[-1]

			pmvt = mvt
			def mvt(y):
				# Ouptut needs room for radius part
				xe = np.empty(jshape[1], dtype=self.centers.dtype)
				# Fill in spatial (and maybe speed) part
				xe[:-1] = pmvt(y)
				# Fill in the radius part
				xe[-1] = np.sum(y)
				return xe

		jac = LinearOperator(shape=jshape, matvec=mv,
				rmatvec=mvt, dtype=self.centers.dtype)
		return jac


	def newton(self, times, pos=None, c=1.5, r=0., maxit=100, tol=1e-6, itargs={}):
		'''
		Use Newton-Raphson iteration to recover the positions of
		unknown element associated with the provided round-trip arrival
		times that correspond to known element positions configured for
		the object instance.

		Initial estimates pos may be specified, but are assumed to be
		zero by default. The rank-2 array pos should specifies the
		estimated coordinates of one element along each row.

		The assumed wave speed is provided in c, while the assumed
		target radius is provided in r. These values may be adjusted by
		optimization according to self.optc or self.optr, respectively.

		Iterations will stop after maxit iterations or when the norm of
		a computed update is less than tol times the norm of the guess
		used to produce the update, whichever occurs first.

		The inverse of the Jacobian is computed using the LSMR
		algorithm (scipy.sparse.linalg.lsmr). The dictionary itargs is
		passed to the LSMR function as its kwargs.

		If self.optc and self.optr are both False, the return value is
		an array of target centers with shape (ntarg, ndim), where
		ntarg is the number of rows in times and ndim is the
		dimensionality of each element in self.centers.

		If at least one of self.optc and self.optr is True, the return
		value will be a tuple whose first element is the
		above-described array of target centers. If self.optc is True,
		the next return element will be the optimized sound speed. If
		self.optr is True, the final return element will be the
		optimized target radius.
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
			raise TypeError('Expected pos.shape to be (%d, %d), not %s' % (times.shape[0], self.centers.shape[1], pos.shape))

		n = nelts * ndim

		for i in range(maxit):
			# Build the Jacobian and right-hand side
			jac = self.jacobian(pos, times)
			cost = self.cost(pos, times, c, r)
			# Use LSMR to invert the system
			delt = lsmr(jac, cost, **itargs)[0]
			# Check for convergence
			conv = (la.norm(delt[:n]) < tol * la.norm(pos))
			# Add the update and break when converged
			pos -= delt[:n].reshape((nelts, ndim), order='C')
			# Include speed and radius updates as appropriate
			if self.optc: c -= delt[n]
			if self.optr: r -= delt[-1]
			if conv: break

		ret = (pos,)
		if self.optc: ret = ret + (c,)
		if self.optr: ret = ret + (r,)

		return ret[0] if len(ret) == 1 else ret


class PlaneTrilateration(MultiPointTrilateration):
	'''
	Use the fixed positions of known elements, embedded in a medium of
	known wave speed, to determine the unknown position of a collection of
	targets using the measured arrival times of signals reflected from the
	known elements. As additional constraints, the positions of the
	recovered elements will be constrained to approximately lie on a plane.
	'''
	def __init__(self, *args, **kwargs):
		'''
		Create a PlaneTrilateration instance using a collection of
		known elements. The arguments are the same as those in the
		MultiPointTrilateration constructor with the addition of an
		optional 'planewt' argument. The argument should be a
		floating-point value (default: 1.0) that represents a weight
		used to scale the coplanarity constraints with respect to the
		MultiPointTrilateration arrival-time constraints. Make planewt
		greater than unity to more strongly enforce the coplanarity
		requirements and less than unity to relax the coplanarity
		requirements.

		If provided, planewt shoud be the fourth (and final) positional
		argument if it is not a keyword argument.
		'''
		# Consume subclass-specific 'planewt' argument
		if len(args) > 4:
			raise TypeError('Too many positional arguments')
		elif len(args) == 4:
			if 'planewt' in kwargs:
				raise TypeError('Duplicate value for argument "planewt"')
			self.planewt = args[3]
			args = args[:3]
		else: self.planewt = kwargs.pop('planewt', 1.0)
		# Finish initialization
		super(PlaneTrilateration, self).__init__(*args, **kwargs)


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

		# Capture a current view of the plane weight
		planewt = self.planewt

		# Build the MVP and its adjoint for a LinearOperator
		def mv(x):
			y = np.empty((jshape[0],), dtype=self.centers.dtype)
			# Compute the per-element MVP
			y[:bjshape[0]] = bjac.matvec(x)
			# Reshape the element coordinates for convenience
			x = np.reshape(x[:nelts*ndim], (nelts, ndim), order='C')
			# Compute the coplanarity parts (weight as desired)
			relx = planewt * (x - np.mean(x, axis=0))
			y[bjshape[0]:] = np.dot(relx, normal)
			return y

		def mvt(y):
			ntrilat = nelts * nrows
			# Compute the independent trilateration contribution
			x = bjac.rmatvec(y[:bjshape[0]])
			# Include coplanarity contribution (weight as desired)
			yplan = y[bjshape[0]:,np.newaxis]
			rely = planewt * (yplan - np.mean(yplan))
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
		cfunc[n:] = self.planewt * np.dot(relpos, normal)
		return cfunc
