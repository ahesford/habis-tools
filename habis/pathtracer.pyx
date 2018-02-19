'''
Routines for tracing paths through and computing arrival times for slowness
images.
'''
# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

import numpy as np
cimport numpy as np

import math
from math import fsum
from numpy.linalg import norm

from .habiconf import HabisConfigParser, HabisConfigError

from libc.stdlib cimport rand, RAND_MAX, malloc, free

from pycwp.cytools.boxer import Box3D
from pycwp.cytools.ptutils cimport *
from pycwp.cytools.boxer cimport Box3D
from pycwp.cytools.interpolator cimport Interpolator3D
from pycwp.cytools.quadrature cimport Integrable, IntegrableStatus

ctypedef enum WNErrCode:
	OK=0,
	BINARY_SEARCH_FAILED,
	FUNCTION_VANISHES,
	NORMAL_VANISHES,
	UNABLE_TO_CACHE

cdef long findrnrm(double *nrms, unsigned long n, double u) nogil:
	'''
	Find the index I of the nearest refrence normal, where normals are
	stored sequentially in the length-(4 * n) array as a sequence

		nrms[4*I:4*(I+1)] = [un, nx, ny, nz],

	such that nrms[4*I] is the largest value un that does not exceed u. The
	array nrms should be sorted in increasing order by the values un.

	If n <= 0 or nrms[0] > u, -1 is returned. Otherwise, a value in the
	range [0, n) will be returned.
	'''
	if n < 1: return -1

	cdef long first, last, middle, idx

	first = 0
	last = n - 1

	# Check the last value first for convenience
	if nrms[4 * last] < u: return last

	# Perform a binary search
	middle = (first + last) / 2
	while first <= last:
		idx = 4 * middle
		if nrms[idx] < u:
			first = middle + 1
		else: last = middle - 1
		middle = (first + last) / 2

	# First should point to the first value not less than u
	return first - 1


cdef long shiftnrm(double *nrms, unsigned long n) nogil:
	'''
	For an array of (4 * n) doubles nrms, shift the right half of the array
	to the left so that

		nrms[4*I:4*(I+1)] = nrms[4*(I+n/2+(n%2)):4*(I+1+n/2+(n%2)]

	for I in [0, n/2).

	The value n/2 is returned.
	'''
	if n < 2: return 0

	cdef long off, i, n2, i4, is4

	n2 = n / 2

	# Find the offset and copy the range, remembering 4 values per index
	off = 4 * (n2 + (n % 2))
	for i in range(4 * n2):
		nrms[i] = nrms[i + off]

	return n2


cdef inline double randf() nogil:
	'''
	Return a sample of a uniform random variable in the range [0, 1].
	'''
	return <double>rand() / <double>RAND_MAX

ctypedef struct PathIntContext:
	point a, b
	bint dograd

ctypedef struct WaveNormIntContext:
	point a, b, h
	long nmax, n, cycles, bad_resets
	WNErrCode custom_retcode
	double normwt
	double minaxdp
	double *normals

class TraceError(Exception): pass

cdef class WavefrontNormalIntegrator(Integrable):
	'''
	A class that holds a reference to an Interpolator3D instance and can
	integrate over straight-ray paths through the grid represented by the
	instance, compensating the integral by tracking the wavefront normal.
	'''
	cdef readonly Interpolator3D data

	@classmethod
	def errmsg(cls, IntegrableStatus code, WNErrCode subcode):
		'''
		Override Integrable.errmsg to process custom codes.
		'''
		if code != IntegrableStatus.CUSTOM_RETURN:
			return super(WavefrontNormalIntegrator, cls).errmsg(code)

		return {
				WNErrCode.BINARY_SEARCH_FAILED: 'Invalid index in binary search',
				WNErrCode.FUNCTION_VANISHES: 'Function vanishes at evaluation point',
				WNErrCode.NORMAL_VANISHES: 'Wavefront normal vanishes at evaluation point',
				WNErrCode.UNABLE_TO_CACHE: 'Cannot add entry to cache',
			}.get(subcode, 'Unknown error')


	def __init__(self, Interpolator3D data):
		'''
		Create a new WavefrontNormalIntegrator to evaluate integrals
		through the interpolated function represented by data.

		The value normwt is used to alter the contribution of slowness
		gradients to the progression on wavefront normals. Set to unity
		to use full compensation or zero to eliminate wavefront-normal
		compensation.
		'''
		self.data = data

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.cdivision(True)
	cdef IntegrableStatus integrand(self, double *results, double u, void *ctx) nogil:
		'''
		Override Integrable.integrand to evaluate the value of the
		interpolated function self.data at a fractional point u along
		the segment from ctx.a to ctx.b (with ctx as a
		WaveNormIntContext struct), with and without a correction based
		on the angle between the estimated wavefront normal and the
		straight-ray path.

		The value ctx.minaxdp is updated with the minimum of its prior
		value and the value of the inner product between the wavefront
		normal and propagation direction as computed by this evaluation
		of the integrand.

		The output array results must have length 2.
		'''
		cdef:
			WaveNormIntContext *wctx = <WaveNormIntContext *>ctx
			point p, gf, nrm, refnrm, ba
			double fv, refu, L, rn, wdp
			double *rnp
			long uidx

		# A context is necessary for integration
		if wctx == <WaveNormIntContext *>NULL:
			return IntegrableStatus.INTEGRAND_MISSING_CONTEXT

		# Compute direction vector (scale grid coordinates to real)
		ba = axpy(-1, wctx.a, wctx.b)
		iptmpy(wctx.h, &ba)
		L = ptnrm(ba)

		if almosteq(L, 0.0):
			# For a zero-length interval, just evaluate at the start
			if not self.data._evaluate(&fv, <point *>NULL, wctx.a):
				return IntegrableStatus.INTEGRAND_EVALUATION_FAILED
			results[0] = results[1] = fv
			return IntegrableStatus.OK

		iscal(1 / L, &ba)

		# Find the reference for wavefront normal tracking
		uidx = findrnrm(wctx.normals, wctx.n, u)
		if uidx == -1:
			if wctx.cycles:
				# This reset is undesirable
				wctx.bad_resets += 1
			# Reset reference to start of path
			refu = 0.
			refnrm = ba
		elif uidx < wctx.nmax:
			# Use cached value as reference
			rnp = &(wctx.normals[4 * uidx])
			refu = rnp[0]
			refnrm = packpt(rnp[1], rnp[2], rnp[3])
		else:
			wctx.custom_retcode = WNErrCode.BINARY_SEARCH_FAILED
			return IntegrableStatus.CUSTOM_RETURN

		# Evaluate the integrand at the pont of interest
		p = lintp(u, wctx.a, wctx.b)
		if not self.data._evaluate(&fv, &gf, p):
			return IntegrableStatus.INTEGRAND_EVALUATION_FAILED

		# Scale gradient properly for wavefront normal tracking
		if almosteq(fv, 0.0):
			wctx.custom_retcode = WNErrCode.FUNCTION_VANISHES
			return IntegrableStatus.CUSTOM_RETURN
		iptdiv(wctx.h, &gf)
		iscal(1 / fv, &gf)

		# Update the wavefront normal
		# TODO: Should this be gf at refu instead of u?
		nrm = axpy(wctx.normwt * L * (u - refu) / dot(ba, refnrm), gf, refnrm)
		rn = ptnrm(nrm)
		if almosteq(rn, 0.0):
			wctx.custom_retcode = WNErrCode.NORMAL_VANISHES
			return IntegrableStatus.CUSTOM_RETURN
		iscal(1 / rn, &nrm)

		# Update the minimum axis dot product
		wdp = dot(nrm, ba)
		wctx.minaxdp = min(wctx.minaxdp, wdp)

		# Scale integrand by compensating factor
		results[0] = fv * wdp
		results[1] = fv

		# Add the new normal to the cache
		uidx += 1
		if uidx >= wctx.nmax:
			# Cache too large, shift second half
			uidx = wctx.n = shiftnrm(wctx.normals, wctx.n)
			wctx.cycles += 1
			# Something went wrong; zero-length cache?
			if uidx >= wctx.nmax:
				wctx.custom_retcode = WNErrCode.UNABLE_TO_CACHE
				return IntegrableStatus.CUSTOM_RETURN
		rnp = &(wctx.normals[4 * uidx])
		rnp[0] = u
		pt2arr(&(rnp[1]), nrm)
		wctx.n = uidx + 1

		return IntegrableStatus.OK


	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.cdivision(True)
	def pathint(self, a, b, double atol, double rtol, h=1.0,
			double normwt=1.0, double maxangle=-1.0,
			unsigned int ncache=512, int reclimit=-1):
		'''
		Given a 3-D path from point a to point b, in grid coordinates,
		use an adaptive Gauss-Kronrod quadrature of order 15 to
		integrate the image associated with self.data along the path,
		with a correction based on tracking of wavefront normals.
		The correction term will be scaled by the value of normwt,
		which must be nonnegative.

		The argument h may be a scalar float or a 3-D sequence of
		floats that defines the grid spacing in world Cartesian
		coordinates. If h is scalar, it is interpreted as [h, h, h]. If
		h is a sequence of three floats, its values define the scaling
		in x, y and z, respectively.

		If maxangle is nonnegative, it specifies the maximum allowable
		angle, in degrees, between the wavefront normal and the
		propagation direction. If the wavefront normal exceeds this
		limit at any evaluation point, the returned value of the
		compensated path integral will be None.

		To evaluate the compensation term in the integrand, estimates
		of the wavefront normal are continually updated with a
		correction that arises from the most recent (best) estimate
		at the nearest evaluation point that precedes the evaluation
		point currently under consideration. This cache requires a
		cache, with a size configurable by ncache. When the cache
		fills, normals for early evaluation points (those nearest to
		point a) will be discarded to make room for later points. When
		a current evaluation point comes before all evaluation points
		in the cache, the "best" estimate is "reset" by reverting to
		the wavefront normal (which is coincident with the straight-ray
		path) at point a. If this reset occurs after early cache values
		have been discarded, accuracy may suffer and a warning will be
		printed. Increase the size of ncache to mitigate this issue.

		The value of reclimit, when nonnegative, limits the number of
		times the path will be subdivided to seek convergence of the
		path integral. If reclimit is negative, no limit applies.

		The return value is a tuple (I1, I2), where I1 is the
		compensated integral (or None, if any maximum angle is exceed)
		and I2 is the uncompensated straight-ray integral.
		'''
		cdef:
			WaveNormIntContext ctx
			double ivals[2]
			double L
			point bah
			IntegrableStatus rcode

		# Initialize the integration context
		tup2pt(&(ctx.a), a)
		tup2pt(&(ctx.b), b)

		try:
			nh = len(h)
		except TypeError:
			# A single scalar applies to all axes
			ctx.h.x = ctx.h.y = ctx.h.z = float(h)
		else:
			# Axes scales are defined independently
			if nh != 3:
				raise ValueError('Argument "h" must be a scalar or 3-sequence')
			ctx.h.x, ctx.h.y, ctx.h.z = float(h[0]), float(h[1]), float(h[2])

		if normwt < 0:
			raise ValueError('Argument "normwt" must be nonnegative')
		ctx.normwt = normwt

		# Storage for normal tracking
		ctx.nmax = ncache
		ctx.n = ctx.cycles = ctx.bad_resets = 0
		ctx.custom_retcode = WNErrCode.OK
		if ctx.nmax > 0:
			ctx.normals = <double *>malloc(4 * ctx.nmax * sizeof(double))
			if ctx.normals == <double *>NULL:
				raise MemoryError('Cannot allocate storage for normal tracking')
		else: ctx.normals = <double *>NULL

		# Initialize the minimum wavefront-propagation dot product
		ctx.minaxdp = 1

		# Integrate and free normal storage
		rcode = self.gausskron(ivals, 2, atol, rtol,
				0., 1., reclimit, <void *>(&ctx))
		free(ctx.normals)

		if rcode != IntegrableStatus.OK:
			errmsg = self.errmsg(rcode, ctx.custom_retcode)
			raise ValueError('Integration failed with error "%s"' % (errmsg,))

		if ctx.bad_resets:
			import warnings
			warnings.warn('Waveform tracking unexpectedly reset '
					'%d times; increase ncache to mitigate' % (ctx.bad_resets,))

		# Scale coordinate axes
		bah = axpy(-1.0, ctx.b, ctx.a)
		iptmpy(ctx.h, &bah)
		L = ptnrm(bah)

		# Make sure angular limit not exceeded for compensated integrals
		if maxangle < 0 or ctx.minaxdp >= np.cos(np.pi * maxangle / 180):
			Ic = ivals[0] * L
		else: Ic = None

		# Scale integrals properly
		return Ic, ivals[1] * L


cdef class PathIntegrator(Integrable):
	'''
	A class that holds a reference to an Interpolator3D instance and can
	integrate over paths through the grid represented by the instance.
	'''
	cdef readonly Interpolator3D data

	def __init__(self, Interpolator3D data):
		'''
		Create a new PathIntegrator to evaluate integrals through the
		interpolated function represented by data.
		'''
		self.data = data

	@cython.wraparound(False)
	@cython.boundscheck(False)
	def _pathopt(self, points, atol, rtol, h, raise_on_fail, warn_on_fail, **kwargs):
		'''
		A helper function for self.minpath to find an optimal path by
		perturbing all coordinates in the N-by-3 array points (except
		for the first and last points) to minimize the path integral
		with self.pathint.

		If scipy.optimize.fmin_l_bfgs_b emits a 'warnflag', this method
		will raise a TraceError if raise_on_fail is True or else will
		issue a warning if warn_on_fail is True.

		Returns the optimum path and its path integral.
		'''
		from scipy.optimize import fmin_l_bfgs_b as bfgs
		import warnings

		points = np.asanyarray(points)
		n, m = points.shape
		if m != 3 or n < 3:
			raise ValueError('Array "points" must have shape (N, 3) with N > 2')

		# Find the optimal path and unflatten the array
		xopt, nf, info = bfgs(self.pathint, points, fprime=None,
					args=(atol, rtol, h, True), **kwargs)
		xopt = xopt.reshape((n, m), order='C')

		if info['warnflag']:
			msg = (f'Optimizer ({n-1} segs, {info["funcalls"]} '
					f' fcalls {info["nit"]} iters) warns ')
			if info['warnflag'] == 1:
				msg += 'limits exceeded'
			elif info['warnflag'] == 2:
				msg += str(info.get('task', 'unknown warning'))
			if raise_on_fail: raise TraceError(msg)
			elif warn_on_fail: warnings.warn(msg)

		return xopt, nf


	@cython.wraparound(False)
	@cython.boundscheck(False)
	def minpath(self, start, end, double atol, double rtol, 
			unsigned long nmax, double ptol, h=1.0,
			double perturb=0.0, unsigned long nstart=1,
			bint warn_on_fail=True, bint raise_on_fail=False, **kwargs):
		'''
		Given 3-vectors start and end in grid coordinates, search for a
		path between start and end that minimizes the path integral of
		the function interpolated by self.data.

		The path will be iteratively divided into at most N segments,
		where N = 2**M * nstart for the smallest integer M that is not
		less than nmax. With each iteration, an optimal path is sought
		by minimizing the object self.pathint(path, atol, rtol, h) with
		respect to all points along the path apart from the fixed
		points start and end. The resulting optimal path is subdivided
		for the next iteration by inserting points that coincide with
		the midpoints of all segments in the currently optimal path.
		Iterations terminate early when the objective value changes by
		less than ptol between two successive optimizations.

		If perturb is greater than zero, the coordinates of the
		midpoints introduced in each iteration will be perturbed by a
		uniformly random variable in the interval [-perturb, perturb]
		before the refined path is optimized.

		If warn_on_fail is True, a warning will be issued whenever an
		optimum path cannot be found for some iteration. Optimization
		will continue despite the failure. If raise_on_fail is True, a
		TraceError exception will be raised whenever an optimum path
		cannot be found for some iteration. Optimization will not
		continue.

		The method scipy.optimize.fmin_l_bfgs_b will be used to
		minimize the objective for each path subdivision. All extra
		kwargs will be passed to fmin_l_bfgs_b. The keyword arguments
		'func', 'x0', 'args', 'fprime' and 'approx_grad' are forbidden.

		The return value will be an L-by-3 array of points (in grid
		coordinates) for some L that describe a piecewise linear
		optimal path, along with the value of the path integral over
		that path.
		'''
		# Validate end points
		cdef point pt
		tup2pt(&pt, start)
		start = pt2tup(pt)
		tup2pt(&pt, end)
		end = pt2tup(pt)

		if nstart < 1: raise ValueError('Value of nstart must be positive')

		# Find the actual maximum number of segments
		cdef unsigned long p = nstart, n, nnx, i, i2, im, im2
		while 0 < p < nmax:
			p <<= 1
		if p < 1: raise ValueError('Value of nmax is out of bounds')

		# Make sure the optimizer is available
		from scipy.optimize import fmin_l_bfgs_b as bfgs
		import warnings

		cdef double[:,:] points, pbest, npoints

		cdef double lf, nf, bf

		# Start with the desired number of segments
		points = np.zeros((nstart + 1, 3), dtype=np.float64, order='C')
		for i in range(0, nstart + 1):
			lf = <double>i / <double>nstart
			bf = 1.0 - lf
			points[i, 0] = bf * start[0] + lf * end[0]
			points[i, 1] = bf * start[1] + lf * end[1]
			points[i, 2] = bf * start[2] + lf * end[2]
		n = nstart

		# Compute the starting cost (and current best)
		if n > 2:
			points, bf = self._pathopt(points, atol, rtol, h,
					raise_on_fail, warn_on_fail, **kwargs)
			pbest = points
			lf = bf
		else:
			pbest = points
			bf = lf = self.pathint(points, atol, rtol, h, False)

		# Double perturbation length for a two-sided interval
		if perturb > 0: perturb *= 2

		while n < p:
			# Interpolate the path
			nnx = n << 1
			npoints = np.zeros((nnx + 1, 3), dtype=np.float64, order='C')

			# Copy the starting point
			npoints[0,0] = points[0,0]
			npoints[0,1] = points[0,1]
			npoints[0,2] = points[0,2]

			# Copy remaining points and interpolate segments
			for i in range(1, n + 1):
				i2 = 2 * i
				im2 = i2 - 1
				im = i - 1
				# Copy the point
				npoints[i2, 0] = points[i, 0]
				npoints[i2, 1] = points[i, 1]
				npoints[i2, 2] = points[i, 2]
				# Compute a midpoint for the expanded segment
				npoints[im2, 0] = 0.5 * (points[im, 0] + points[i, 0])
				npoints[im2, 1] = 0.5 * (points[im, 1] + points[i, 1])
				npoints[im2, 2] = 0.5 * (points[im, 2] + points[i, 2])

				if perturb > 0:
					# Sample in [-0.5, 0.5] to perturb midpoints
					npoints[im2, 0] += perturb * (randf() - 0.5)
					npoints[im2, 1] += perturb * (randf() - 0.5)
					npoints[im2, 2] += perturb * (randf() - 0.5)

			n = nnx
			points = npoints

			# Optimize the interpolated path
			points, nf = self._pathopt(points, atol, rtol, h,
					raise_on_fail, warn_on_fail, **kwargs)

			if nf < bf:
				# Record the current best path
				bf = nf
				pbest = points

			# Check for convergence
			if abs(nf - lf) < ptol: break

		# Return the best points and the path integral
		return np.asarray(pbest), bf


	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.cdivision(True)
	def pathint(self, points, double atol, double rtol,
			h=1.0, bint grad=False, bint gk=False, int reclimit=-1):
		'''
		Given control points specified as rows of an N-by-3 array of
		grid coordinates, use an adaptive quadrature to integrate the
		image associated with self.data along the piecewise linear path
		between the points.

		As a convenience, points may also be a 1-D array of length 3N
		that represents the two-dimensional array of points flattened
		in C order.

		The argument h may be a scalar float or a 3-D sequence of
		floats that defines the grid spacing in world Cartesian
		coordinates. If h is scalar, it is interpreted as [h, h, h]. If
		h is a sequence of three floats, its values define the scaling
		in x, y and z, respectively.

		If grad is False, only the integral will be returned. If grad
		is True, the return value will be (ival, igrad), where ival is
		the path integral and igrad is an N-by-3 array wherein igrad[i]
		is the gradient of ival with respect to points[i]. By
		convention, igrad[0] and igrad[N - 1] are identically zero. If
		the input array points was a 1-D flattened version of points,
		the output igrad will be similarly flattened in C order.

		If gk is True, adaptive Gauss-Kronrod quadrature will be used
		to compute path integrals. Otherwise, adaptive Simpson
		quadrature (which re-uses function evaluations at sub-interval
		endpoints and may be more efficient) will be used.

		The value of reclimit, when nonnegative, limits the number of
		times the path will be subdivided to seek convergence of the
		path integral. If reclimitis negative, no limit applies.

		If the second dimension of points does not have length three,
		or if any control point falls outside the interpolation grid,
		a ValueError will be raised.
		'''
		cdef bint flattened = False
		# Make sure points is a well-behaved array
		points = np.asarray(points, dtype=np.float64)

		# For convenience, handle a flattened array of 3-D points
		if points.ndim == 1:
			flattened = True
			points = points.reshape((-1, 3), order='C')
		elif points.ndim != 2:
			raise ValueError('Points must be a 1-D or 2-D array')

		cdef unsigned long npts = points.shape[0]
		if points.shape[1] != 3:
			raise ValueError('Length of second dimension of points must be 3')

		cdef:
			double ival = 0.0, fval = 0.0
			double[:,:] pts = points
			point scale
			point lgrad
			point rgrad

			double results[7]
			double ends[14]
			unsigned int nval
			PathIntContext ctx
			point bah

			IntegrableStatus rcode

			double L

			unsigned long i, im1

		cdef np.ndarray[np.float64_t, ndim=2] pgrad

		try:
			nh = len(h)
		except TypeError:
			# A single scalar applies to all axes
			scale.x = scale.y = scale.z = float(h)
		else:
			# Axes scales are defined independently
			if nh != 3:
				raise ValueError('Argument "h" must be a scalar or 3-sequence')
			scale.x, scale.y, scale.z = float(h[0]), float(h[1]), float(h[2])

		# Initialize the integration context
		ctx.dograd = grad

		# Allocate output for the path gradient
		if ctx.dograd:
			pgrad = np.zeros((npts, 3), dtype=np.float64)
			nval = 7
		else:
			pgrad = None
			nval = 1

		# Initialize the left point
		ctx.a = packpt(pts[0,0], pts[0,1], pts[0,2])
		# The integrand ignores ctx.b when u is 0
		ctx.b = ctx.a
		# Evaluate the integrand at the left endpoint if needed
		if not gk:
			rcode = self.integrand(ends, 0., <void *>(&ctx))
			if rcode != IntegrableStatus.OK:
				errmsg = self.errmsg(rcode)
				raise ValueError('Integrand evaluation failed with message "%s"' % (errmsg,))

		for i in range(1, npts):
			# Initialize the right point
			ctx.b = packpt(pts[i,0], pts[i,1], pts[i,2])
			# Calculate integrals over the segment
			if not gk:
				# Evalute integrand at left endpoint
				rcode = self.integrand(&(ends[nval]), 1., <void *>&(ctx))
				if rcode != IntegrableStatus.OK:
					errmsg = self.errmsg(rcode)
					raise ValueError('Integrand evaluation failed with message "%s"' % (errmsg,))
				rcode = self.simpson(results, nval, atol, rtol,
						reclimit, <void *>(&ctx), ends)
				if rcode != IntegrableStatus.OK:
					errmsg = self.errmsg(rcode)
					raise ValueError('Simpson integration failed with message "%s"' % (errmsg,))
			else:
				rcode = self.gausskron(results, nval, atol, rtol,
						0., 1., reclimit, <void *>(&ctx))
				if rcode != IntegrableStatus.OK:
					errmsg = self.errmsg(rcode)
					raise ValueError('Gauss-Kronrod integration failed with message "%s"' % (errmsg,))

			# Scale coordinate axes
			bah = axpy(-1.0, ctx.b, ctx.a)
			iptmpy(scale, &bah)
			L = ptnrm(bah)

			# Scale integrals properly
			ival = L * results[0]
			if ctx.dograd:
				lgrad = packpt(results[1], results[2], results[3])
				rgrad = packpt(results[4], results[5], results[6])
				# Scale gradients by inverse coordinate factors
				iptdiv(scale, &lgrad)
				iptdiv(scale, &rgrad)
				# Start with integrals of function gradients
				iscal(L, &lgrad)
				iscal(L, &rgrad)
				# Now add the step integral
				iaxpy(results[0] / L, bah, &lgrad)
				iaxpy(-results[0] / L, bah, &rgrad)

			# Add segment contribution to path integral
			fval += ival

			# Cycle endpoints and integrands for next round
			ctx.a = ctx.b
			for im1 in range(nval):
				ends[im1] = ends[im1 + nval]

			if not grad: continue

			# Add contribution to gradient from segment endpoints
			im1 = i - 1
			pgrad[im1, 0] += lgrad.x
			pgrad[im1, 1] += lgrad.y
			pgrad[im1, 2] += lgrad.z
			pgrad[i, 0] += rgrad.x
			pgrad[i, 1] += rgrad.y
			pgrad[i, 2] += rgrad.z

		# Return just the function, if no gradient is desired
		if not grad: return fval

		# Force endpoint gradients to zero
		pgrad[0,0] = pgrad[0,1] = pgrad[0,2] = 0.0
		im1 = npts - 1
		pgrad[im1,0] = pgrad[im1,1] = pgrad[im1,2] = 0.0

		# Return the function and the (flattened, if necessary) gradient
		if flattened: return fval, pgrad.ravel('C')
		else: return fval, pgrad


	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.cdivision(True)
	cdef IntegrableStatus integrand(self, double *results, double u, void *ctx) nogil:
		'''
		Override Integrable.integrand to evaluate the value of the
		interpolated function self.data at a point along the segment
		from ctx.a to ctx.b (with ctx interpreted as a PathIntContext
		struct), along with (if ctx.dograd) the gradient contributions
		with respect to each endpoint of the segment.

		The output array results must have length 1 if ctx.dograd is
		False, or length 7 if ctx.dograd is True.
		'''
		cdef:
			PathIntContext *sctx = <PathIntContext *>ctx
			point p
			point gf
			double fv

		# Cannot integrate without context!
		if sctx == <PathIntContext *>NULL:
			return IntegrableStatus.INTEGRAND_MISSING_CONTEXT

		# Find the evaluation point
		p = lintp(u, sctx.a, sctx.b)

		if sctx.dograd:
			if not self.data._evaluate(&fv, &gf, p):
				return IntegrableStatus.INTEGRAND_EVALUATION_FAILED
			mu = 1 - u
			results[0] = fv
			pt2arr(&(results[1]), scal(1 - u, gf))
			pt2arr(&(results[4]), scal(u, gf))
		else:
			if not self.data._evaluate(&fv, <point *>NULL, p):
				return IntegrableStatus.INTEGRAND_EVALUATION_FAILED
			results[0] = fv

		return IntegrableStatus.OK


class PathTracer(object):
	'''
	A class that encapsulates a 3-D gridded slowness image and allows
	determination of an optimal propagation path.
	'''
	@classmethod
	def fromconf(cls, config):
		'''
		Create a PathTracer instance from config, an instance of
		habis.habiconf.HabisConfigParser. The parser should contain a
		'pathtrace' section which defines the parameters of the
		PathTracer instance. The options in the 'pathtrace' section
		are:

		'grid' (required, no default):

		  A dictionary of the form

		    { 'lo': [lx,ly,lz], 'hi': [hx,hy,hz], 'ncell': [nx,ny,nz] }

		  that defines 'lo', 'hi', and 'ncell' PathTracer init arguments.

		'segmax' (default: 256):
		'ptol' (default: 1e-3):
		'atol' (default: 1e-6):
		'rtol' (default: 1e-4):

		  Floats that specify values for corresponding PathTracer init
		  arguments.

		'bropts' (default: { }):
		'sropts' (default: { }):

		  Dictionaries that provide keyword arguments for corresponding
		  PathTracer init arguments.
		'''

		# Use this section for path tracing
		psec = 'pathtrace'

		def _throw(msg, e):
			errmsg = msg + ' in [' + psec + ']'
			raise HabisConfigError.fromException(errmsg, e)

		# Verify that no extra configuration options are specified
		try: keys = config.keys(psec)
		except Exception as e: _throw('Unable to read configuration', e)

		extra_keys = keys.difference({'prefilter', 'defval',
			'interpolator', 'grid', 'atol', 'rtol', 'bropts', 'sropts'})
		if extra_keys:
			first_key = next(iter(extra_keys))
			raise HabisConfigError(f'Unrecognized key {first_key} '
						f'in configuration section [{psec}]')

		try:
			grid = config.get(psec, 'grid', mapper=dict, checkmap=False)
			ncell = grid.pop('ncell')
			lo = grid.pop('lo')
			hi = grid.pop('hi')
			if grid: raise KeyError('Invalid keyword ' + str(next(iter(grid))))
		except Exception as e: _throw('Configuration must specify valid grid', e)

		try: atol = config.get(psec, 'atol', mapper=float, default=1e-6)
		except Exception as e: _throw('Invalid optional atol', e)

		try: rtol = config.get(psec, 'rtol', mapper=float, default=1e-4)
		except Exception as e: _throw('Invalid optional atol', e)

		try:
			bropts = config.get(psec, 'bropts',
					default={ }, mapper=dict, checkmap=False)
		except Exception as e: _throw('Invalid optional bropts', e)

		try:
			sropts = config.get(psec, 'sropts',
					default={ }, mapper=dict, checkmap=False)
		except Exception as e: _throw('Invalid optional sropts', e)

		try: defval = config.get(psec, 'defval', mapper=float, default=None)
		except Exception as e: _throw('Invalid optional defval', e)

		try: interpolator = config.get(psec, 'interpolator')
		except Exception as e: _throw('Configuration must specify valid interpolator', e)

		try:
			prefilter = config.get(psec, 'prefilter',
					default=None, mapper=dict, checkmap=False)
		except Exception as e: _throw('Invalid optional prefilter', e)

		# Create the instance
		return cls(lo, hi, ncell, atol, rtol,
				interpolator, defval, prefilter, bropts, sropts)


	def __init__(self, lo, hi, ncell, atol, rtol, interpolator,
			defval=None, prefilter=None, bropts={ }, sropts={ }):
		'''
		Define a grid (lo x hi) subdivided into ncell voxels on which a
		slowness map should be interpreted, along with options for
		tracing straight-ray or bent-ray paths through slowness models.

		The parameters atol and rtol specify, respectively, absolute
		and relative integration tolerances for both straight-ray and
		bent-ray integration methods.

		For bent-ray tracing tracing, PathIntegrator.minpath is used to
		find an optimal path. Keyword arguments for the method
		PathIntegrator.minpath may be provided in the dictionary
		bropts; bropts must contain at least the "nmax" and "ptol"
		keys if bent-ray tracing is to be used.

		For straight-ray tracing, WavefrontNormalIntegrator.pathint is
		used to compute compensated and uncompensated straight-ray
		integrals. Optional keyword arguments to the method
		WavefrontNormalIntegrator.pathint may be provided in the
		dictionary sropts.

		For integration, a 3-D Numpy array is interpolated after an
		optional prefiltering step according to the function built by
		self.make_interpolator(interpolator, defval, prefilter). The
		filter and its name are stored as self.interpolator and
		self.prefilter, respectively.
		'''
		# Define the grid for images
		self.box = Box3D(lo, hi, ncell)

		# Copy integration parameters
		self.atol = float(atol)
		self.rtol = float(rtol)

		# Copy optional argument dicts
		self.bropts = dict(bropts)
		self.sropts = dict(sropts)

		# Build the array interpolator
		intp, fname = self.make_interpolator(interpolator, defval, prefilter)
		self.interpolator, self.prefilter = intp, fname

		# By default, there is no slowness
		self._slowness = None

	@staticmethod
	def make_interpolator(interpolator, defval=None, prefilter=None):
		'''
		For the given interpolator, which names an Interpolator3D
		subclass in the module pycwp.cytools.interpolator, and a
		pre-filter argument prefilter, return a function that will
		compute an Interpolator3D for a 3-D Numpy array. The
		interpolator will have a defval value defval, and prefilter is
		used to control optional filtering applied before path
		integrals are evaluated. Also returned is a string indicating
		the type of pre-filtering selected, or None if prefilter is
		None.

		If prefilter is not None, it must be a dictionary with a
		'method' key. The method key is interpreted in one of two ways.
		If prefilter inclues a 'parfilter' key, the 'method' key will
		be passed to habis.mpfilter.parfilter to produce a parallel
		version of the named filter. The value for 'parfilter' can
		either be True, which will pass None to parfilter as the comm
		argument, or it can be an MPI communicator object that will be
		passed as the comm argument.

		Without a 'parfilter' key (or if the 'parfilter' value
		evaluates to False), the 'method' key represents the name of a
		filter function in scipy.ndimage or, if none exists there, in
		pycwp.filter.

		The prefilter dictionary may contain an optional 'args'
		sequence (default: ()) and an optional 'kwargs' dictionary
		(default: {}) that will be passed to the filter function. As a
		convenience, if args is a scalar, it will be wrapped in a
		tuple.

		If prefilter is None, no pre-filtering is done.

		Hence, a function f returned by this function satisfies:

		* When prefilter is a dictionary,

		  f(s) = interpolator(filter(s, *args, **kwargs), defval),

		  where filter is the function indicated by the 'method' key,
		  and args and kwargs are associated with respective keys in
		  prefilter.

		* When prefilter is None, f(s) = interpolator(s, defval).
		'''
		# Pull the interpolator by name
		import pycwp.cytools.interpolator
		interpolator = getattr(pycwp.cytools.interpolator, interpolator)

		# Read the filter
		try:
			tfname = prefilter['method']
		except KeyError:
			raise KeyError('Dict "prefilter" must contain "method" key')
		except TypeError:
			if prefilter is not None:
				raise TypeError('Argument "prefilter" must be None or a dictionary')
			# With no prefilter, just build the interpolator
			def filterfunc(s):
				return interpolator(s, defval)
			tfname = None
		else:
			# Check for positional arguments
			args = prefilter.get('args', ())

			# Check for keyword arguments
			kwargs = prefilter.get('kwargs', { })

			pcomm = prefilter.get('parfilter', False)

			if pcomm:
				from .mpfilter import parfilter
				if pcomm is True:
					filt = parfilter(tfname)
					tfname = f'parfilter({tfname})'
				else:
					filt = parfilter(tfname, comm=pcomm)
					tfname = f'parfilter({tfname}, comm={pcomm})'
			else:
				try:
					import scipy.ndimage
					filt = getattr(scipy.ndimage, tfname)
					tfname = f'scipy.ndimage.{tfname}'
				except (AttributeError, ImportError):
					import pycwp.filter
					filt = getattr(pycwp.filter, tfname)
					tfname = f'pycwp.filter.{tfname}'

			try: len(args)
			except TypeError: args = (args,)

			def filterfunc(s):
				# Do the filtering before
				ns = filt(s, *args, **kwargs)
				return interpolator(ns, defval)

		return filterfunc, tfname


	def set_slowness(self, s):
		'''
		If s is not None, store self.interpolator(s) as the slowness
		model through which self.trace will evaluate path integrals.

		If s is None, remove the stored interpolator.
		'''
		if s is None: self._slowness = None
		else: self._slowness = self.interpolator(s)


	def get_slowness(self):
		'''
		Return a reference to the stored slowness.
		'''
		return self._slowness


	def pathmap(self, points, fresnel=0):
		'''
		For an L-by-3 array (or compatible sequence) or points in world
		coordinates, march the path through self.box to produce and
		return a map (i, j, k) -> L, where (i, j, k) is a cell index in
		self.box and L is the accumulated length of the path defined by
		the points.

		If fresnel is nonzero, it must be positive and will define the
		radius of the Fresnel ellipsoid (possibly bent, if points
		contains more than two points) into which the path will be
		expanded.
		'''
		fresnel = float(fresnel or 0)
		if fresnel < 0:
			raise ValueError('Argument "fresnel" must be nonnegative')

		points = np.asarray(points, dtype=np.float64)
		if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
			raise ValueError('Argument "points" must have shape (L, 3), L >= 2')

		if fresnel:
			# Trace the Fresnel zone through the box
			plens = self.box.fresnel(points, fresnel)

			# Compute the total path length
			tlen = fsum(norm(r - l) for r, l in zip(points, points[1:]))

			# Convert Fresnel-zone weights to integral contributions
			wtot = fsum(plens.values())
			return { k: tlen * v / wtot for k, v in plens.items() }

		# March the points, ensure single-segment march still a list
		marches = self.box.raymarcher(points)
		if points.shape[0] < 3: marches = [marches]

		# Accumulate the length of each path in each cell
		plens = { }
		for (st, ed), march in zip(zip(points, points[1:]), marches):
			# Compute whol length of this path segment
			dl = norm(ed - st)
			for cell, (tmin, tmax) in march.items():
				# Convert fractional length to real length
				# 0 <= tmin <= tmax <= 1 guaranteed by march algorithm
				contrib = (tmax - tmin) * dl

				# Add the contribution to the list for this cell
				try: plens[cell].append(contrib)
				except KeyError: plens[cell] = [contrib]

		# Accumulate the contributions to each cell
		# Return the path map and the path-length integral
		return { k: fsum(v) for k, v in plens.items() }


	def trace(self, src, rcv, fresnel=0, intonly=False, mode='bent'):
		'''
		Given a 3-D Numpy array s, a source with world coordinates src
		and a receiver with world coordinates rcv, compute a path
		integral from src to rcv using, when mode is 'bent',

		  PathIntegrator(si).minpath(gs, gr, self.atol,
				self.rtol, h=self.box.cell, **self.bropts),

		or, when mode is 'straight',

		  WavefrontNormalIntegrator(si).pathint(gs, gr, self.atol,
				self.rtol, h=self.box.cell, **self.sropts),

		where si is the interpolator returned by self.get_slowness, and
		gs and gr are grid coordinates of src and rcv, respectively,
		according to self.box.cart2cell.

		If self.get_slowness() returns None, this method will fail with
		a TypeError.

		Note that, in 'bent' mode, the mandatory parameters 'ptol' and
		'nmax' of PathIntegrator.minpath are required to be in
		self.bropts as keyword arguments.

		In 'bent' mode, the path integral is a scalar value: the
		integral from src to rcv through an optimized path. If fresnel
		is a positive numeric value, rays in 'bent' mode will be
		expanded into ellipsoids that represent the first Fresnel zone
		for a dominant wavelength specified as the value of fresnel, in
		the same units as self.box.cell.

		In 'straight' mode, the path integral is a 2-tuple (c, s),
		where c is the wavefront-compensated path integral and s is the
		uncompensated straight-ray integral. Compensated straight-ray
		integrals are incompatible with Fresnel ellipsoids, so c always
		behaves as if fresnel were zero; the uncompensated integral
		will use Fresnel ellipsoids if fresnel is nonzero.

		If intonly is True, only the path integral(s) will be returned.
		Otherwise, the path will be marched through self.box to produce
		a map (i, j, k) -> L, where (i, j, k) is a cell index in
		self.box and L is the accumulated length of the optimum path
		through that cell. The return value in this case will be
		(pathmap, points, pathint), where pathmap is the cell-to-length
		map, points is an N-by-3 array containing the world coordinates
		of the N (>= 2) points that define the traced path and pathint
		is the single-value or double-value path integral.

		Any Exceptions raised by WavefrontNormalIntegrator.pathint or
		PathIntegrator.minpath will not be caught by this method.
		'''
		# Grab the interpolator
		si = self.get_slowness()
		if si is None:
			raise TypeError('Call set_slowness() before calling trace()')

		# Build the integrator for the image and verify shape
		mode = str(mode).lower()
		if mode == 'bent': integrator = PathIntegrator(si)
		elif mode == 'straight': integrator = WavefrontNormalIntegrator(si)
		else: raise ValueError('Argument "mode" must be "bent" or "straight"')

		box = self.box
		if integrator.data.shape != box.ncell:
			raise ValueError('Shape of si must be %s' % (box.ncell,))

		# Convert world coordinates to grid coordinates
		gsrc = box.cart2cell(*src)
		grcv = box.cart2cell(*rcv)

		fresnel = max(fresnel or 0, 0)

		if mode == 'bent':
			# Use the bent-ray tracer to find an optimal path
			points, pint = integrator.minpath(gsrc, grcv,
					self.atol, self.rtol, h=box.cell,
					raise_on_fail=True, **self.bropts)
			# Convert path to world coordinates for marching
			points = np.array([box.cell2cart(*p) for p in points])
		else:
			# Do path integration only without Fresnel zones
			pint = integrator.pathint(gsrc, grcv, self.atol,
					self.rtol, h=box.cell, **self.sropts)
			# Straight path just has a start and end
			points = np.array([src, rcv], dtype=np.float64)

		# If only the integral is desired, just return it
		if intonly and not fresnel: return pint

		# Trace the path for returning
		plens = self.pathmap(points, fresnel)

		if fresnel:
			# Reintegrate over Fresnel path
			fint = fsum(v * si.evaluate(*k, grad=False)
						for k, v in plens.items())
			# Replace relevant integral with Fresnel integral
			if mode == 'bent': pint = fint
			else: pint = (pint[0], fint)

			if intonly: return pint

		return plens, points, pint


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def srcompensate(trpairs, elements, Box3D bx, Interpolator3D si, long N):
	'''
	For a given list of (t, r) pairs trpairs that describe transmit-receive
	indices into elements, a coordinate grid defined by the Box3D bx, an
	interpolated slowness image si (as an Interpolator3D instance) defined
	on the grid, and a desired quadrature order N, return a list of tuples
	(Ic, Is, dl, rn, errc, errs) for each entry in trpairs, where Ic is the
	compensated straight-ray arrival time, Is is the uncompensated
	straight-ray arrival time, dl is the unit direction of the straight-ray
	path from elements[t] to elements[r], rn is the final wavefront normal
	at the receiver and err and errs are, respectively, error estimates for
	the integrals Ic and Is.
	'''
	cdef:
		long i, Ne
		double sv, lu, ru, ndl, kwt, gwt, kval, gval, ksval, gsval, nln, L
		point svg, tx, rx, tg, rg, dl, ln, x
		double[:,:] weights

	# Verify grid
	if bx.ncell != si.shape:
		raise ValueError('Grid of bx and si must match')

	# Build the quadrature weights, converting intervals
	from pycwp.cytools.quadrature import Integrable
	gkwts = [(0.5 * (1. - nd), 0.5 * kw, 0.5 * gw)
			for nd, kw, gw in Integrable.gkweights(N, 1e-8)]
	gkwts.extend((1. - nd, kw, gw) for nd, kw, gw in reversed(gkwts[:N]))

	weights = np.asarray(gkwts, dtype=np.float64)
	Ne = weights.shape[0]

	# Trace the paths one-by-one
	results = [ ]
	for t, r in trpairs:
		kval, gval, ksval, gsval = 0., 0., 0., 0.

		# Find element coordinates and convert to grid coordinates
		tup2pt(&tx, elements[t])
		tup2pt(&rx, elements[r])
		tg = bx._cart2cell(tx.x, tx.y, tx.z)
		rg = bx._cart2cell(rx.x, rx.y, rx.z)

		# Find unit direction and length of path
		dl = axpy(-1., tx, rx)
		L = ptnrm(dl)
		if almosteq(L, 0.0):
			raise ValueError('Path (%d, %d) is degenerate' % (t, r))
		iscal(1 / L, &dl)

		# Initial wavefront normal is direction vector
		lu = 0.
		ln = dl

		# First evaluation point is transmitter
		x = tg
		if not si._evaluate(&sv, &svg, x):
			raise ValueError('Cannot evaluate integrand at %s' % (pt2tup(x),))
		if almosteq(sv, 0.0):
			raise ValueError('Integrand vanishes at %s' % (pt2tup(x),))
		# Scale gradient properly
		iptdiv(bx._cell, &svg)
		iscal(1 / sv, &svg)
		# Find wavefront "drift"
		ndl = dot(ln, dl)

		for i in range(Ne):
			if almosteq(ndl, 0.0):
				raise ValueError('Wavefront normal is orthogonal to path at %s' % (pt2tup(x),))
			# Next evaluation point
			ru = weights[i,0]
			kwt = weights[i,1]
			gwt = weights[i,2]
			# New wavefront normal (renormalized) and "drift"
			iaxpy((ru - lu) * L / ndl, svg, &ln)
			nln = ptnrm(ln)
			if almosteq(nln, 0.0):
				raise ValueError('Wavefront normal vanishes at %s' % (pt2tup(x),))
			iscal(1 / nln, &ln)
			ndl = dot(ln, dl)
			# New evaluation point and function evaluation
			x = lintp(ru, tg, rg)
			if not si._evaluate(&sv, &svg, x):
				raise ValueError('Cannot evaluate integrand at %s' % (pt2tup(x),))
			if almosteq(sv, 0.0):
				raise ValueError('Integrand vanishes at %s' % (pt2tup(x),))
			iptdiv(bx._cell, &svg)
			iscal(1 / sv, &svg)
			# Update total integrals
			kval += kwt * ndl * sv
			ksval += kwt * sv
			gval += gwt * ndl * sv
			gsval += gwt * sv
			lu = ru
		results.append((L * kval, L * ksval, pt2tup(dl),
			pt2tup(ln), L * abs(gval - kval), L * abs(gsval - ksval)))
	return results
