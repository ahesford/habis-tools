'''
Routines for tracing paths through and computing arrival times for slowness
images.
'''
# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

#cython: embedsignature=True

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
	double *normals

class TraceError(Exception): pass

cdef class WavefrontNormalIntegrator(Integrable):
	'''
	A class that holds a reference to an Interpolator3D instance and can
	integrate over straight-ray paths through the grid represented by the
	instance, compensating the integral by tracking the wavefront normal.
	'''
	cdef readonly Interpolator3D data
	cdef public double normwt

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


	def __init__(self, Interpolator3D data, double normwt=1.0):
		'''
		Create a new WavefrontNormalIntegrator to evaluate integrals
		through the interpolated function represented by data.

		The value normwt is used to alter the contribution of slowness
		gradients to the progression on wavefront normals. Set to unity
		to use full compensation or zero to eliminate wavefront-normal
		compensation.
		'''
		self.data = data
		if normwt < 0 or normwt > 1:
			raise ValueError('Value of normwt must be in range [0, 1]')
		self.normwt = normwt

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

		The output array results must have length 2.
		'''
		cdef:
			WaveNormIntContext *wctx = <WaveNormIntContext *>ctx
			point p, gf, nrm, refnrm, ba
			double fv, refu, L, rn
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
		nrm = axpy(self.normwt * L * (u - refu) / dot(ba, refnrm), gf, refnrm)
		rn = ptnrm(nrm)
		if almosteq(rn, 0.0):
			wctx.custom_retcode = WNErrCode.NORMAL_VANISHES
			return IntegrableStatus.CUSTOM_RETURN
		iscal(1 / rn, &nrm)

		# Scale integrand by compensating factor
		results[0] = fv * dot(nrm, ba)
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
	def pathint(self, a, b, double atol, double rtol,
			h=1.0, unsigned int ncache=512):
		'''
		Given a 3-D path from point a to point b, in grid coordinates,
		use an adaptive Gauss-Kronrod quadrature of order 15 to
		integrate the image associated with self.data along the path,
		with a correction based on tracking of wavefront normals.

		The argument h may be a scalar float or a 3-D sequence of
		floats that defines the grid spacing in world Cartesian
		coordinates. If h is scalar, it is interpreted as [h, h, h]. If
		h is a sequence of three floats, its values define the scaling
		in x, y and z, respectively.

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

		The return value is a tuple (I1, I2), where I1 is the
		compensated integral and I2 is the uncompensated straight-ray
		integral.
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

		# Storage for normal tracking
		ctx.nmax = ncache
		ctx.n = ctx.cycles = ctx.bad_resets = 0
		ctx.custom_retcode = WNErrCode.OK
		if ctx.nmax > 0:
			ctx.normals = <double *>malloc(4 * ctx.nmax * sizeof(double))
			if ctx.normals == <double *>NULL:
				raise MemoryError('Cannot allocate storage for normal tracking')
		else: ctx.normals = <double *>NULL

		# Integrate and free normal storage
		rcode = self.gausskron(ivals, 2, atol, rtol, 0., 1., <void *>(&ctx))
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

		# Scale integrals properly
		return ivals[0] * L, ivals[1] * L


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
	def minpath(self, start, end, unsigned long nmax,
			double atol, double rtol, double ptol,
			h=1.0, double perturb=0.0, unsigned long nstart=1,
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
			xopt, nf, info = bfgs(self.pathint, points, fprime=None,
						args=(atol, rtol, h, True), **kwargs)
			points = xopt.reshape((n + 1, 3), order='C')

			if info['warnflag']:
				msg = 'Optimizer (%d segs, %d fcalls, %d iters) warns ' % (n, info['funcalls'], info['nit'])
				if info['warnflag'] == 1:
					msg += 'limits exceeded'
				elif info['warnflag'] == 2:
					msg += str(info.get('task', 'unknown warning'))
				if raise_on_fail: raise TraceError(msg)
				elif warn_on_fail: warnings.warn(msg)

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
			h=1.0, bint grad=False, bint gk=False):
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
				rcode = self.simpson(results, nval, atol, rtol, <void *>(&ctx), ends)
				if rcode != IntegrableStatus.OK:
					errmsg = self.errmsg(rcode)
					raise ValueError('Simpson integration failed with message "%s"' % (errmsg,))
			else:
				rcode = self.gausskron(results, nval, atol, rtol, 0., 1., <void *>(&ctx))
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
		'itol' (default: 1e-5):

		  Floats that specify values for corresponding PathTracer init
		  arguments.

		'optimizer' (default: { }):

		  A dictionary that provides keyword arguments to the method
		  pycwp.cytool.sinterpolator.Interpolator3D.minpath.
		'''

		# Use this section for path tracing
		psec = 'pathtrace'

		def _throw(msg, e):
			errmsg = msg + ' in [' + psec + ']'
			raise HabisConfigError.fromException(errmsg, e)

		try:
			grid = config.get(psec, 'grid', mapper=dict, checkmap=False)
			ncell = grid.pop('ncell')
			lo = grid.pop('lo')
			hi = grid.pop('hi')
			if grid: raise KeyError('Invalid keyword ' + str(next(iter(grid))))
		except Exception as e: _throw('Configuration must specify valid grid', e)

		try: ptol = config.get(psec, 'ptol', mapper=float, default=1e-3)
		except Exception as e: _throw('Invalid optional ptol', e)

		try:
			itol = config.getlist(psec, 'itol',
					mapper=float, default=1e-5, checkmap=False)
		except Exception as e: _throw('Invalid optional itol', e)

		try: segmax = config.get(psec, 'segmax', mapper=int, default=256)
		except Exception as e: _throw('Invalid optional segmax', e)

		try:
			optargs = config.get(psec, 'optimizer',
					default={ }, mapper=dict, checkmap=False)
		except Exception as e: _throw('Invalid optional optimizer', e)

		# Create the instance
		return cls(lo, hi, ncell, segmax, itol, ptol, optargs)


	def __init__(self, lo, hi, ncell, segmax, itol, ptol, optargs={ }):
		'''
		Define a grid (lo x hi) subdivided into ncell voxels on which a
		slowness map should be interpreted, along with options for
		tracing optimal paths through the (variable) slowness.

		The parameters segmax, ptol, itol are stored to be passed as
		the nmax, itol and ptol arguments, respectively, to the method
		pycwp.cytools.interpolator.Interpolator3D.minpath for an
		instance of Interpolator3D passed to a later call to the method
		self.pathtrace. The optargs dictionary holds additonal keyword
		arguments to be passed to minpath.
		'''
		# Define the grid for images
		self.box = Box3D(lo, hi, ncell)

		# Copy adaptive parameters
		self.ptol = float(ptol)
		# Treat integration tolerance as a list, if possible
		try: self.itol = [ float(iv) for iv in itol ]
		except TypeError: self.itol = [ float(itol), 0. ]
		if len(self.itol) != 2:
			raise ValueError('Parameter "itol" must be scalar or length-2 sequence')
		self.nmax = int(segmax)

		# Copy the optimization arguments
		self.optargs = dict(optargs)


	def trace(self, si, src, rcv, fresnel=0, intonly=False):
		'''
		Given an interpolatored slowness map si (as a 3-D Numpy array
		or pycwp.cytools.interpolator.Interpolator3D), a source with
		world coordinates src and a receiver with world coordinates
		rcv, trace an optimum path from src to rcv using

		  PathIntegrator(si).minpath(gs, gr, self.nmax,
				self.itol[0], self.itol[1], self.ptol, 
				self.box.cell, **self.optargs),

		where gs and gr are grid coordinates of src and rcv,
		respectively, according to self.box.cart2cell.

		If fresnel is a positive numeric value, rays will be expanded
		into ellipsoids that represent the first Fresnel zone for a
		dominant wavelength specified as the value of fresnel, in the
		same units as self.box.cell.

		If intonly is True, only the integral of the slowness over the
		optimum path will be returned. Otherwise, the optimum path will
		be marched through self.box to produce a map (i, j, k) -> L,
		where (i, j, k) is a cell index in self.box and L is the
		accumulated length of the optimum path through that cell. The
		return value in this case will be (pathmap, pathint), where
		pathmap is this cell-to-length map and pathint is the
		integrated slowness over the optimum path.

		Any Exceptions raised by PathIntegrator.minpath will not be
		caught by this method.
		'''
		box = self.box

		# Build the integrator for the image and verify shape
		integrator = PathIntegrator(si)
		if integrator.data.shape != box.ncell:
			raise ValueError('Shape of si must be %s' % (box.ncell,))

		# Convert world coordinates to grid coordinates
		gsrc = box.cart2cell(*src)
		grcv = box.cart2cell(*rcv)

		# Use preconfigured options to evaluate minimum path
		popt, pint = integrator.minpath(gsrc, grcv, self.nmax,
				self.itol[0], self.itol[1], self.ptol,
				box.cell, raise_on_fail=True, **self.optargs)

		# If only the integral is desired, just return it
		if intonly and fresnel <= 0: return pint

		# Convert path to world coordinates for marching
		points = np.array([box.cell2cart(*p) for p in popt])

		if (fresnel or 0) > 0:
			# Trace the Fresnel zone through the box
			plens = box.fresnel(points, fresnel)

			# Compute the total length of the path
			tlen = fsum(norm(r - l) for r, l in zip(points, points[1:]))

			# Convert Fresnel-zone weights to integral contributions
			wtot = fsum(plens.values())
			plens = { k: tlen * v / wtot for k, v in plens.items() }

			# Reintegrate over Fresnel path
			pint = fsum(v * si.evaluate(*k, grad=False)
					for k, v in plens.items())

			return pint if intonly else (plens, pint)

		marches = box.raymarcher(points)

		# Make sure single-segment march still a list
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
		return { k: fsum(v) for k, v in plens.items() }, pint


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
