'''
Routines for tracing paths through and computing arrival times for slowness
images.
'''
# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys

import cython
cimport cython

import numpy as np
cimport numpy as np

import warnings

import math
from math import fsum
from numpy.linalg import norm

from .habiconf import HabisConfigParser, HabisConfigError, watchConfigErrors

from libc.stdlib cimport rand, RAND_MAX, malloc, free
from libc.math cimport sqrt, pi, fabs

from pycwp.cytools.boxer import Box3D, Segment3D
from pycwp.cytools.ptutils cimport *
from pycwp.cytools.boxer cimport Box3D, Segment3D
from pycwp.cytools.interpolator cimport Interpolator3D
from pycwp.cytools.quadrature cimport Integrable, IntegrableStatus

def _path_as_array(path):
	'''
	Convert path, which should either be a Segment3D or an array-compatible
	object, to an (L, 3) array of float values.
	'''
	if isinstance(path, Segment3D):
		return np.asarray([path.start, path.end], dtype='float64')
	else:
		path = np.asarray(path, dtype='float64')
		if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3:
			raise ValueError('Shape of path must be (L >= 2, 3)')
		return path


cdef inline double sqr(double x) nogil:
	return x * x

cdef extern from "complex.h":
	double complex cexp(double complex)
	double complex csinh(double complex)
	double cimag(double complex)

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
	def _pathopt(self, points, atol, rtol, h, damp,
			raise_on_fail, warn_on_fail, **kwargs):
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

		points = np.asanyarray(points)
		n, m = points.shape
		if m != 3 or n < 3:
			raise ValueError('Array "points" must have shape (N, 3) with N > 2')

		# Find the optimal path and unflatten the array
		xopt, nf, info = bfgs(self.pathint, points, fprime=None,
					args=(atol, rtol, h, True, damp), **kwargs)
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

		# Re-evaluate the integral if damping was used for optimization
		if (damp or 0) and damp > 0:
			nf = self.pathint(xopt, atol, rtol, h, False)

		return xopt, nf


	@cython.wraparound(False)
	@cython.boundscheck(False)
	@staticmethod
	def interpolate_path(path, unsigned long nstart):
		'''
		Given path as an array (or compatible) with shape (L >= 2, 3)
		and a desired number of starting segments nstart, attempt to
		interpolate the path to nstart segments.

		If nstart == 0, no interpolation is performed, and the path is
		returned as if nstart == L - 1.

		If L == 2, the path will be interpolated by equally
		distributing (nstart + 1) points along the linear segment from
		path[0] to path[1].

		If L > 2, no interpolation is performed regardless of the value
		of nstart. A warning will be issued if nstart != L - 1.

		A ValueError will be raised if path has the wrong shape or
		nstart is 0.
		'''
		cdef:
			double[:,:] dpath, points
			long L

		path = np.asarray(path, dtype=np.float64)
		if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3:
			raise ValueError('Argument "path" must have shape (L >= 2, 3)')

		dpath = path
		L = dpath.shape[0] - 1

		if nstart < 1: nstart = L

		if L > 1:
			if nstart != L:
				msg = f'{L} -> {nstart} interpolation is not supported'
				warnings.warn(msg)
			return path

		# No need to interpolate if one segment is desired
		if L == nstart: return path

		# Interpolate the input segment as desired
		points = np.zeros((nstart + 1, 3), dtype=np.float64, order='C')
		for i in range(0, nstart + 1):
			lf = <double>i / <double>nstart
			bf = 1.0 - lf
			points[i, 0] = bf * dpath[0,0] + lf * dpath[1,0]
			points[i, 1] = bf * dpath[0,1] + lf * dpath[1,1]
			points[i, 2] = bf * dpath[0,2] + lf * dpath[1,2]

		return np.asarray(points)


	@cython.wraparound(False)
	@cython.boundscheck(False)
	def minpath(self, path, double atol, double rtol, double ptol,
			unsigned long nmax, h=1.0, double damp=0.0,
			double perturb=0.0, unsigned long nstart=0,
			bint warn_on_fail=True, bint raise_on_fail=False, **kwargs):
		'''
		Given path, an array-compatible sequence with shape (L >= 2, 3)
		that describes the nodes of a piecwise linear path in grid
		coordinates, search for an optimal path between start and end
		that minimizes the path integral of the function interpolated
		by self.data. The path is first interpolated (if appropriate)
		and the value of nstart will be replaced by by calling

			path = self.interpolate_path(path, nstart)
			nstart = path.shape[0] - 1

		This path will be iteratively divided into at most N segments,
		where N = 2**M * nstart for the smallest integer M that is not
		less than nmax.

		With each iteration, an optimal path is sought
		by minimizing the object

			self.pathint(path, atol, rtol, h, damp=damp)

		with respect to all points along the path apart from the fixed
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
		# Find the actual maximum number of segments
		cdef:
			unsigned long p, n, nnx, i, i2, im, im2
			double[:,:] points, pbest, npoints
			cdef double lf, nf, bf

		# Make sure the optimizer is available
		from scipy.optimize import fmin_l_bfgs_b as bfgs

		# Interpolate the path as desired and note number of segments
		points = self.interpolate_path(path, nstart)
		n = p = points.shape[0] - 1

		while 0 < p < nmax: p <<= 1
		if p < 1: raise ValueError('Value of nmax is out of bounds')

		# Compute the starting cost (and current best)
		if n > 2:
			points, bf = self._pathopt(points, atol, rtol, h,
					damp, raise_on_fail, warn_on_fail, **kwargs)
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
					damp, raise_on_fail, warn_on_fail, **kwargs)

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
			h=1.0, bint grad=False, double damp=0.0,
			bint gk=False, int reclimit=-1):
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

		If damp is greater than 0, the path integrand will be replaced
		by (self.data + damp), which effectively adds a length penalty
		when the path integral is optimized. Otherwise, damp is
		ignored.

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

		# Ignore negative damping parameters
		if damp < 0: damp = 0

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

			# Include damping parameter
			results[0] += damp

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

			if not ctx.dograd: continue

			# Add contribution to gradient from segment endpoints
			im1 = i - 1
			pgrad[im1, 0] += lgrad.x
			pgrad[im1, 1] += lgrad.y
			pgrad[im1, 2] += lgrad.z
			pgrad[i, 0] += rgrad.x
			pgrad[i, 1] += rgrad.y
			pgrad[i, 2] += rgrad.z

		# Return just the function, if no gradient is desired
		if not ctx.dograd: return fval

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


class AbstractPathTracer(object):
	'''
	An abstract class that encapsulates a 3-D gridded slowness image and
	allows integrating paths through the slowness.
	'''
	habisConfigSectionName = 'pathtrace'

	@classmethod
	def fromconf(cls, config):
		'''
		Create an AbstractPathTracer instance from config, an instance of
		habis.habiconf.HabisConfigParser. The parser will be converted
		to an instance and kwargs dictionary using cls.kwargsFromConfig
		and instantiated by calling subcls(**kwargs), where subcls is
		the class returned by

		  getattr(module, instance.title() + 'PathTracer'),

		where module is the current module.
		'''
		# Try to determine an instance in the configuration
		psec = cls.habisConfigSectionName
		with watchConfigErrors('instance', psec):
			instance = config.get(psec, 'instance', default='', mapper=str)
			subcls = instance.title() + 'PathTracer'
			subcls = getattr(sys.modules[__name__], subcls)

		try: keys = config.keys(psec)
		except Exception as e:
			msg = f'Unable to read keys in config section {psec}'
			raise HabisConfigError.fromException(msg, e)

		# Parse arguments for the instance
		kwargs = subcls.kwargsFromConfig(config)

		# Keys "instance" and "grid" are handled specially
		extra_keys = keys.difference({'instance', 'grid'}.union(kwargs))
		if extra_keys:
			msg = (f'Unprocessed key "{next(iter(extra_keys))}" '
					f'in configuration section "{psec}"')
			raise HabisConfigError(msg)

		# Create the instance
		return subcls(**kwargs)


	@classmethod
	def kwargsFromConfig(cls, config):
		'''
		From config, an instance of habis.habiconf.HabisConfigParser,
		build a kwargs dictionary that maps values in the configuration
		file to constructor arguments for this class.
		'''
		psec = cls.habisConfigSectionName

		kwargs = { }

		with watchConfigErrors('grid', psec):
			grid = config.get(psec, 'grid', mapper=dict, checkmap=False)
			for kw in ['ncell', 'lo', 'hi']:
				kwargs[kw] = grid.pop(kw)
			if grid: raise KeyError(f'Invalid keyword {next(iter(grid))}')

		with watchConfigErrors('defval', psec, False):
			kwargs['defval'] = config.get(psec, 'defval', mapper=float)

		with watchConfigErrors('interpolator', psec):
			kwargs['interpolator'] = config.get(psec, 'interpolator')

		with watchConfigErrors('prefilter', psec, False):
			kwargs['prefilter'] = config.get(psec, 'prefilter',
							mapper=dict, checkmap=False)

		return kwargs


	def __init__(self, lo, hi, ncell, interpolator, defval=None, prefilter=None):
		'''
		Define a grid (lo x hi) subdivided into ncell voxels on which a
		slowness map should be interpreted.

		For integration, a 3-D Numpy array is interpolated after an
		optional prefiltering step according to the function built by
		self.make_interpolator(interpolator, defval, prefilter). The
		filter and its name are stored as self.interpolator and
		self.prefilter, respectively.
		'''
		# Define the grid for images
		self.box = Box3D(lo, hi, ncell)

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
		if s is None:
			self._slowness = None
		else:
			slowness = self.interpolator(s)
			if slowness.shape != self.box.ncell:
				raise ValueError('Shape of slowness must equal self.box.ncell')
			self._slowness = slowness


	def get_slowness(self):
		'''
		Return a reference to the stored slowness.
		'''
		return self._slowness


	def pathmap(self, path):
		'''
		Using the pathmarch() function, march the provided path through
		self.box and return a map (i, j, k) -> L, where (i, j, k) is an
		index into self.box and L is the length of the path (over all
		segments) through that cell.
		'''
		# March the path, forcing the output to be a list of maps
		path = _path_as_array(path)
		marches = pathmarch(path, self.box, force_list=True)

		# Accumulate the length of each path in each cell
		plens = { }
		for (st, ed), march in zip(zip(path, path[1:]), marches):
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


	def trace(self, path, intonly=False):
		'''
		Perform a simple integral through self.get_slowness(), which
		must not be None, over the given path as a Segment3D or an
		array-like object of shape (L >= 2, 3). The integral is a
		simple product of the operator implied by self.pathmap and the
		slowess.

		If intonly is True, the integral is returned alone. Othewise,
		the return value will be (pathmap, integral), where pathmap is
		the map returned by self.pathmap(path).
		'''
		si = self.get_slowness()
		if si is None:
			raise TypeError('Call set_slowness() before calling trace()')

		pmap = self.pathmap(path)
		pint = si.wtsum(pmap)
		if intonly: return pint
		return pmap, pint


class PathTracer(AbstractPathTracer):
	'''
	A subclass of AbstractPathTracer to implement tracing along (possibly
	bent) paths through inhomogeneous media, with the option to use Fresnel
	ellipsoids as fat rays.
	'''
	@classmethod
	def kwargsFromConfig(cls, config):
		'''
		Specialize the kwargs dictionary built by the method
		super().kwargsFromConfig(config) to construct arguments for
		instantiation of a PathTracer.
		'''
		psec = cls.habisConfigSectionName

		kwargs = super().kwargsFromConfig(config)

		for tol in ('atol', 'rtol'):
			with watchConfigErrors(tol, psec):
				kwargs[tol] = config.get(psec, tol, mapper=float)

		kws = ('ptol', 'fresnel', 'nmax')
		tps = (float, float, int)
		for kw, typ in zip(kws, tps):
			with watchConfigErrors(kw, psec, False):
				kwargs[kw] = config.get(psec, kw, mapper=typ)

		for kw in ('pathint_opts', 'minpath_opts'):
			with watchConfigErrors(kw, psec, False):
				kwargs[kw] = config.get(psec, kw,
						mapper=dict, checkmap=False)

		return kwargs


	def __init__(self, atol, rtol, ptol=0, nmax=1, fresnel=None,
			pathint_opts={}, minpath_opts={}, *args, **kwargs):
		'''
		Define an AbstractPathTracer as
		
			super().__init__(*args, **kwargs).

		The absolute (atol) and relative (rtol) integration tolerances
		(as positive floats), together with extra keyword arguments in
		pathint_opts, govern the use of PathIntegrator.pathint to
		evaluate integrals through slowness models along paths.

		The parameters ptol and nmax control path optimization when
		tracing. The value of ptol must be a float, while the value of
		nmax must be a positive integer. If ptol <= 0 or nmax == 1, no
		path optimization will be attemped, and path integrals computed
		with self.trace will use paths as defined. Whenever ptol > 0
		and nmax > 1, any input path will first by optimized with
		PathIntegrator.minpath, passing additional keyword arguments
		through minpath_opts.

		An optional Fresnel width fresnel, which must be None or a
		nonnegative integer, allows traced paths to be represented as
		zero-width rays or Fresnel ellipsoids with the associated width
		parameter.
		'''
		# Configure specific options
		tolerances = tuple(float(v) for v in (atol, rtol))
		if any(v <= 0 for v in tolerances):
			raise ValueError('Tolerances "atol" and "rtol" must be positive')
		self.atol, self.rtol = tolerances

		self.ptol = float(ptol)
		if self.ptol < 0:
			raise ValueError('Path tolerance "ptol" must be nonnegative')

		self.nmax = int(nmax)
		if self.nmax < 1:
			raise ValueError('Maximum segment count "nmax" must be positive')

		self.fresnel = float(fresnel or 0)
		if self.fresnel < 0:
			raise ValueError('Fresnel width must be None or nonnegative')

		self.pathint_opts = dict(pathint_opts)
		self.minpath_opts = dict(minpath_opts)

		# Configure common tracer parameters
		super().__init__(*args, **kwargs)


	def pathmap(self, path):
		'''
		If self.fresnel is not positive, return the output of

			super().pathmap(path).

		Otherwise, compute the Fresnel zone for the given path as

			fresnel_zone(path, self.box, self.fresnel)

		and return a map (i, j, k) -> W, where the weight W is the
		total length of the path multiplied by the ratio of the Fresnel
		weight at cell (i, j, k) to the sum of the total Fresnel
		weights in the zone.
		'''
		if self.fresnel > 0:
			# Trace the Fresnel zone through the box
			path = _path_as_array(path)
			plens = fresnel_zone(path, self.box, self.fresnel)

			# Compute the total path length
			tlen = fsum(norm(r - l) for r, l in zip(path, path[1:]))

			# Convert Fresnel-zone weights to integral contributions
			wtot = fsum(plens.values())
			return { k: tlen * v / wtot for k, v in plens.items() }
		else:
			return super().pathmap(path)


	def trace(self, path, intonly=False):
		'''
		Given path, an array-like sequence with shape (L >= 2, 3) or a
		Segment3D instance (which is treated as [path.start, path.end])
		that describes the world coordinates of nodes that define a
		piecewise linear path, evaluate the integral of a slowness
		self.get_slowness() (which must not be None) along the path.

		Where necessary, path integrals are evaluated by an instance

		  pi = PathIntegrator(self.get_slowness()).

		If self.ptol > 0 and self.nmax > 1, the path is first optimized
		by calling

		  pi.minpath(gpoints, self.atol, self.rtol, self.ptol,
		  		self.nmax, h=self.box.cell, **self.minpath_opts),

		where gpoints is the path converted to grid coordinates. Note
		that minpath provides the value of the integral as well as the
		optimal path.

		If path optimization is not used and Fresnel zones are not
		enabled, the path integral is evaluated by calling

		  pi.pathint(gpoints, self.atol, self.rtol,
		  		h=self.box.cell, **self.pathint_opts).

		If self.fresnel is True, the integral of the path (either as
		input or as produced by pi.minpath) is replaced by the product
		of the map self.pathmap(path) and the slowness model.

		If intonly is True, only the integral will be returned.
		Otherwise, the return value will be (pathmap, integral), where
		pathmap = self.pathmap(path).
		'''
		si = self.get_slowness()
		if si is None:
			raise TypeError('Call set_slowness() before calling trace()')

		path = _path_as_array(path)

		# Convert the path to grid coordinates
		box = self.box
		gpts = np.array([box.cart2cell(*p) for p in path], dtype='float64')

		if self.ptol > 0 and self.nmax > 1:
			# Replace input path with optimized path
			integrator = PathIntegrator(si)
			gpts, pint = integrator.minpath(gpts, self.atol,
					self.rtol, self.ptol, self.nmax,
					h=self.box.cell, **self.minpath_opts)
			if intonly: return pint
			path = np.array([box.cell2cart(*p) for p in gpts])
		elif not self.fresnel:
			# Perform path integral when Fresnel zones are not used
			integrator = PathIntegrator(si)
			pint = integrator.pathint(gpts, self.atol, self.rtol,
						h=self.box.cell, **self.pathint_opts)
			if intonly: return pint

		# The pathmap is required
		pmap = self.pathmap(path)

		if self.fresnel:
			# Perform the integral over the Fresnel zone
			pint = si.wtsum(pmap)
			if intonly: return pint

		return pmap, pint


	def compensated_trace(self, path, intonly=False):
		'''
		Trace the given path through self.get_slowness(), which must
		not be None, to return both compensated and uncompensated
		integrals over the path. The integrals are computed as

		  wi = WavefrontNormalIntegrator(self.get_slowness())
		  wi.pathint(gsrc, grcv, self.atol, self.rtol,
		  		h=self.box.cell, **self.pathint_opts),

		where gsrc and grcv are the grid coordinates of the start and
		end of the path.

		The path must be a single Segment3D or array-compatible object
		with shape (2, 3).

		The values of self.ptol, self.nmax and self.fresnel are ignored
		for this method.

		If intonly is True, return (cval, uval), which are the
		compensated and uncompensated path integrals, respectively.
		Otherwise, return ((cval, uval), pathmap), where pathmap is the
		map returned by self.pathmap(path).
		'''
		path = _path_as_array(path)
		if path.shape != (2, 3):
			err = 'Compensated tracing only works for single-segment paths'
			raise NotImplementedError(err)

		gsrc = self.box.cart2cell(*path[0])
		grcv = self.box.cart2cell(*path[1])

		si = self.get_slowness()
		if si is None:
			raise TypeError('Call set_slowness() before calling compensated_trace()')

		integrator = WavefrontNormalIntegrator(si)
		cval, uval = integrator.pathint(gsrc, grcv, self.atol,
				self.rtol, h=self.box.cell, **self.pathint_opts)

		if intonly: return (cval, uval)
		return (cval, uval), self.pathmap(path)


class RytovPathTracer(AbstractPathTracer):
	'''
	A subclass of AbstractPathTracer to implement straight-ray tracing
	using Rytov kernels for fat rays.
	'''
	@classmethod
	def kwargsFromConfig(cls, config):
		'''
		Specialize the kwargs dictionary built by the method
		super().kwargsFromConfig(config) to construct arguments for
		instantiation of a RytovPathTracer.
		'''
		psec = cls.habisConfigSectionName

		kwargs = super().kwargsFromConfig(config)

		for kw in ('l', 's'):
			with watchConfigErrors(kw, psec):
				kwargs[kw] = config.get(psec, kw, mapper=float)

		with watchConfigErrors('rytov_opts', psec, False):
			kwargs['rytov_opts'] = config.get(psec, 'rytov_opts',
							mapper=dict, checkmap=False)

		return kwargs


	def __init__(self, l, s, rytov_opts={}, *args, **kwargs):
		'''
		Define a PathTracer as super().__init__(*args, **kwargs).

		The wavelength l and slowness s, as positive floats, are passed
		as corresponding parameters to rytov_zone to convert a straight
		path to a Rytov zone. Additional rytov_zone keyword arguments
		may be provided in rytov_opts.
		'''
		# Configure specific options
		self.l = float(l)
		if self.l <= 0: raise ValueError('Wavelength "l" must be positive')

		self.s = float(s)
		if self.s <= 0: raise ValueError('Slowness "s" must be positive')

		self.rytov_opts = dict(rytov_opts)

		super().__init__(*args, **kwargs)


	def pathmap(self, path):
		'''
		For a given path, which must be a single Segment3D or an
		array-like object of shape (2, 3), return a map (i, j, k) -> L
		as computed by

		  rytov_zone(path, self.box, self.l, self.s, **self.rytov_opts).
		'''
		return rytov_zone(path, self.box, self.l, self.s, **self.rytov_opts)


@cython.wraparound(False)
@cython.boundscheck(False)
def fresnel_zone(p, Box3D box, double l):
	'''
	Find the first Fresnel zone of a path p, which is either a single
	Segment3D or a 2-D array of shape (N, 3), where N >= 2, that provides
	control points for a piecewise linear curve through a grid defined by
	the Box3D box.

	The Fresnel zone is represented as a map that takes the form

		(i, j, k) -> 1 - R,

	where (i, j, k) is the index of a grid cell within the first Fresnel
	zone and R = D / r, where D is the perpendicular distance from the
	midpoint of the cell to the nearest segment and r is the radius of the
	first Fresnel zone at the projection of the cell midpoint onto the
	nearest segment:

		r = sqrt(l * d1 * d2 / (d1 + d2)),

	where d1 is the distance (along the curve) from the start of the curve
	to the point where the radius is evaluated, d2 is the distance (along
	the curve) between the point where the radius is evaluated and the end
	of the curve, and l is the dominant wavelength.

	A point is considered to be in the first Fresnel zone if R is in the
	range (0, 1].
	'''
	cdef double[:,:] pts

	pts = _path_as_array(p)

	if pts.shape[0] < 2 or pts.shape[1] != 3:
		raise ValueError('Argument "p" must be a Segment3D or have shape (N,3), N >= 2')

	cdef:
		Segment3D seg
		int axis, axp, axpp, k, inzone
		point spt, ept, mpt, mpc
		double smin, scell, rad, tlen, slen, weight
		long stslab, edslab, slab, hcc, hcr
		long i, j, sg, sgp, irad, jrad

		double tlims[2]
		double scl[3]
		double ecl[3]
		double lo[3]
		double hi[3]
		long ccell[3]
		long ncell[3]
		long ngrid[3]

	hits = { }

	# Find total length of curve
	tlen = 0.0
	for sg in range(pts.shape[0] - 1):
		sgp = sg + 1
		tlen += sqrt(sqr(pts[sgp,0] - pts[sg,0]) +
				sqr(pts[sgp,1] - pts[sg,1]) +
				sqr(pts[sgp,2] - pts[sg,2]))

	# Copy the grid to an array
	ngrid[0], ngrid[1], ngrid[2] = box.nx, box.ny, box.nz

	# Loop through each segment
	slen = 0.0
	for sg in range(pts.shape[0] - 1):
		sgp = sg + 1
		seg = Segment3D((pts[sg,0], pts[sg,1], pts[sg,2]),
				(pts[sgp,0], pts[sgp,1], pts[sgp,2]))

		if not Box3D._intersection(tlims,
				box._lo, box._hi, seg._start, seg._end):
			# Segment does not intersect the grid
			continue

		# Find the major axis and the perpendicular axes
		direction = seg.direction
		axis = max(xrange(3), key=lambda i: abs(direction[i]))
		axp = <int>((axis + 1) % 3)
		axpp = <int>((axis + 2) % 3)

		# Find the end points of the intersection
		spt = lintp(max(tlims[0], 0.0), seg._start, seg._end)
		ept = lintp(min(tlims[1], 1.0), seg._start, seg._end)

		# Convert Cartesian points to cell coordinates as arrays
		pt2arr(scl, box._cart2cell(spt.x, spt.y, spt.z))
		pt2arr(ecl, box._cart2cell(ept.x, ept.y, ept.z))

		# Store the corners of slabs as arrays
		pt2arr(lo, box._lo)
		pt2arr(hi, box._hi)

		# Grab the slab minimum, thickness and maximum count
		smin = lo[axis]
		if axis == 0: scell = box._cell.x
		elif axis == 1: scell = box._cell.y
		elif axis == 2: scell = box._cell.z
		else: raise IndexError('Invalid major axis %d' % (axis,))

		# Compute the start and end slabs of the segment
		stslab = max(0, min(ngrid[axis] - 1, <long>(scl[axis])))
		edslab = max(0, min(ngrid[axis] - 1, <long>(ecl[axis])))
		if edslab < stslab: stslab, edslab = edslab, stslab

		# Loop through all slabs to pick up neighbors
		for slab in range(stslab, edslab + 1):
			# Adjust slab corners along slabbed axis
			lo[axis] = smin + scell * slab
			hi[axis] = smin + scell * (slab + 1)
			# Check slab intersections
			if not Box3D._intersection(tlims,
					packpt(lo[0], lo[1], lo[2]),
					packpt(hi[0], hi[1], hi[2]),
					seg._start, seg._end):
				# For some reason, segment misses slab
				continue
			# Clip intersections
			tlims[0], tlims[1] = max(0, tlims[0]), min(1, tlims[1])
			# Find midpoint and its cell coordinates
			mpt = lintp(0.5 * (tlims[0] + tlims[1]), seg._start, seg._end)
			mpc = box._cart2cell(mpt.x, mpt.y, mpt.z)
			# Find cell index for midpoint
			ccell[0] = <long>mpc.x
			ccell[1] = <long>mpc.y
			ccell[2] = <long>mpc.z

			# Search the neighborhood
			ncell[axis] = ccell[axis]
			irad = max(ccell[axp], ngrid[axp] - ccell[axp])
			jrad = max(ccell[axpp], ngrid[axpp] - ccell[axpp])
			for i in range(irad):
				hcr = 0
				for j in range(jrad):
					hcc = 0
					# Loop through all neighbor corners
					for k in range(4):
						# Neighbor cell index
						ncell[axp] = ccell[axp] + i * (2 * (k % 2) - 1)
						ncell[axpp] = ccell[axpp] + j * (2 * (k / 2) - 1)
						# Evaluate midpoint of neighbor
						inzone = fresnel_weight(&weight, box, seg,
									ncell, l, tlen, slen)
						if inzone:
							key = (ncell[0], ncell[1], ncell[2])
							hits[key] = max(hits.get(key, 0.0), weight)
						hcc += inzone
					# Stop if no hits found in column
					if not hcc: break
					# Otherwise, check the next column
					hcr += hcc
				# Stop if no hits found in row
				if not hcr: break
		# Keep track of global length at next segment start
		slen += seg.length
	return hits


def pathmarch(p, Box3D box, double step=REALEPS, bint force_list=False):
	'''
	March along the given p, which is either a single Segment3D instance or
	a 2-D array of shape (N, 3), where N >= 2, that defines control
	points of a piecewise linear curve.

	A march of a single linear segment accumulates a map of the form
	(i, j, k) -> (tmin, tmax), where (i, j, k) is the index of a cell that
	intersects the segment and (tmin, tmax) are the minimum and maximum
	lengths (as fractions of the segment length) along which the segment
	and cell intersect.

	If p is a Segment3D instance, it behaves as if it were an array
	[p.start, p.end]. When the input behaves as an array of shape (2, 3)
	and force_list is False, a single map will be returned. Otherwise, when
	p behaves as an array of shape (N, 3), a list of (N - 1) of maps will
	be returned, with maps[i] providing the intersection map for the
	segment from p[i] to p[i+1].

	As a segment exits each encountered cell, a step along the segment is
	taken to advance into another intersecting cell. The length of the step
	will be, in units of the segment length,

		step * sum(2**i for i in range(q)),

	where the q is chosen at each step as the minimum nonnegative integer
	that guarantees advancement to another cell. Because this step may be
	nonzero, cells which intersect the segment over a total length less
	than step may be excluded from the intersection map.
	'''
	cdef double[:,:] pts
	cdef Segment3D seg
	cdef point s, e

	if isinstance(p, Segment3D):
		seg = <Segment3D>p
		path = segmarch(seg._start, seg._end, box, step)
		if force_list: return [path]
		else: return path

	pts = np.asarray(p, dtype=np.float64)
	if pts.shape[0] < 2 or pts.shape[1] != 3:
		raise ValueError('Argument "p" must have shape (N,3), N >= 2')

	# Capture the start of the first segment
	s = packpt(pts[0,0], pts[0,1], pts[0,2])

	# Accumulate results for multiple segments
	results = [ ]

	cdef unsigned long i
	for i in range(1, pts.shape[0]):
		# Build current segment and march
		e = packpt(pts[i,0], pts[i,1], pts[i,2])
		results.append(segmarch(s, e, box, step))
		# Move end to start for next round
		s = e

	# Special case: a single path returns a single map unless lists are forced
	if pts.shape[0] == 2 and not force_list:
		return results[0]

	return results

cdef object segmarch(point start, point end, Box3D box, double step=REALEPS):
	'''
	For a line segment from a point start to a point end, both in world
	coordinates, return a mapping from cell indices in the grid defined by
	Box3D to a tuple (ts, te) that indicates the fractional length at which
	the segment enters (ts) and exits (te) the cell. Both ts and te will be
	in the range [0, 1].

	The step parameter is interpreted as described in the pathmarch
	docstring.
	'''
	# Make sure the segment intersects this box
	cdef double tlims[2]

	intersections = { }
	if not Box3D._intersection(tlims, box._lo, box._hi, start, end):
		return intersections

	if step <= 0: step = -step

	# Keep track of accumulated and max length
	cdef double t = max(0, tlims[0])
	cdef double tmax = min(tlims[1], 1)
	# This is a dynamically grown step into the next cell
	cdef double cstep

	cdef point lo, hi
	cdef long i, j, k, ni, nj, nk

	# Find the cell that contains the current test point
	box._cellForPoint(&i, &j, &k, lintp(t, start, end))

	while t < tmax:
		box._boundsForCell(&lo, &hi, i, j, k)
		if not Box3D._intersection(tlims, lo, hi, start, end):
			stt = pt2tup(start)
			edt = pt2tup(end)
			raise ValueError(f'Segment {stt} -> {edt} does '
						f'not intersect cell {(i,j,k)}')

		if 0 <= i < box.nx and 0 <= j < box.ny and 0 <= k < box.nz:
			# Record a hit inside the grid
			key = i, j, k
			val = max(0, tlims[0]), min(tlims[1], 1)
			intersections[i,j,k] = val

		# Advance t; make sure it lands in another cell
		t = tlims[1]
		cstep = step
		while t < tmax:
			# Find the cell containing the point
			box._cellForPoint(&ni, &nj, &nk, lintp(t, start, end))
			if i != ni or j != nj or k != nk:
				# Found a new cell; record and move on
				i, j, k = ni, nj, nk
				break
			# Otherwise, stuck in same cell; bump t
			t += cstep
			# Increase step for next time
			cstep *= 2

	return intersections


@cython.cdivision(True)
cdef int fresnel_weight(double *weight, Box3D box, Segment3D seg,
		long *cell, double l, double tlen=-1.0, double slen=-1.0) nogil:
	'''
	For a given segment seg and a cell with indices { i, j, k }, calculate
	the ratio D / r for a perpendicular distance D between the cell
	midpoint and the segment and r is the Fresnel radius for a dominant
	wavelength l evaluated at the projection of the cell midpoint on the
	segment.

	The Fresnel radius is evaluated assuming that seg is a piece of an
	overall piecewise linear curve with length tlen, and seg starts at a
	length slen along the overall curve. If tlen is negative, it will be
	replaced by the value value seg.length. If slen is negative, a value of
	0 will be used.

	A cell midpoint is considered to be in the Fresnel zone if r is
	nonzero, D / r is in the range [0, 1) and the projection of the
	midpoint on seg is within the bounds of the segment.

	If the midpoint is in the Fresnel zone, the value (1 - D / r) will be
	stored in *weight and a 1 will be returned.

	If the midpoint is not in the Fresnel zone, 0 will be returned and
	weight will not be touched.
	'''
	cdef:
		double rad, nval, oval
		double pdist[2]
		point mpt

	if not (0 <= cell[0] < box.nx and
			0 <= cell[1] < box.ny and 0 <= cell[2] < box.nz):
		# Out-of-bounds cells are not in the zone
		return 0

	# Find the midpoint of the cell
	mpt = box._cell2cart(cell[0] + 0.5, cell[1] + 0.5, cell[2] + 0.5)

	# Find the (bounded) distance and the fractional projection
	seg._ptdist(pdist, mpt, 1)

	# Use sensible default values for curve length parameters
	if tlen < 0: tlen = seg.length
	if slen < 0: slen = 0.0

	# Convert fractional distance to real distance along curve, bounded
	pdist[1] = max(min(tlen, seg.length * pdist[1] + slen), 0)

	# Calculate Fresnel radius
	rad = sqrt(l * pdist[1] * (1.0 - pdist[1] / tlen))

	if pdist[0] >= rad or almosteq(rad, 0.0):
		# If point is beyond radius, or radius is zero, not in zone
		return 0

	weight[0] = 1 - pdist[0] / rad
	return 1


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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def descent_walk(Box3D box, start, end, Interpolator3D field,
		unsigned long cycles=1, double step=1.0, double tol=1e-3,
		double c=0.5, double tau=0.5, bint report=False):
	'''
	Perform a steepest-descent walk from a point p through the given field
	as an Interpolator3D instance capable of evaluating both the field, f,
	and its gradient at arbitrary points within the given Box3D box.

	The walk proceeds from some point p to another point q by performing an
	Armijo backtracking line search along the negative gradient of the
	field. Let h = norm(box.cell) be the diameter of a cell in this box, g
	= grad(f)(p) be the gradent of f at p, and m = |g| be the norm of the
	gradient. The search will select the next point q = (p - alpha * g /
	m), where the factor alpha = tau**k * step * h for the smallest integer
	k such that

		f(p - alpha * g / m) - f(p) <= -alpha * c * m.

	The walk terminates when one of:

	1. A point q of the path comes within h of the point end,
	2. The path runs off the edge of the grid,
	3. The factor tau**k < tol when finding the next step, or
	4. The same grid cell is encountered more than cycles times.

	If the argument "report" is True, a second return value, a string with
	value 'destination', 'boundary', 'stationary', or 'cycle' will be
	returned to indicate the relevant termination criterion (1, 2, 3 or 4,
	respectively).

	A ValueError will be raised if the field has the wrong shape.
	'''
	cdef:
		long nx, ny, nz
		point p, t, gf, pd, np
		long i, j, k, ti, tj, tk
		double fv, tv, m, lim, alpha
		unsigned long hc

	# Make sure the field is double-array compatible
	nx, ny, nz = field.shape
	if (nx != box.nx or ny != box.ny or nz != box.nz):
		raise ValueError('Shape of field must be %s' % (box.ncell,))

	# Make sure the provided start and end points are valid
	tup2pt(&p, start)
	tup2pt(&t, end)

	# Convert start and end points to grid coordinates
	p = box._cart2cell(p.x, p.y, p.z)
	t = box._cart2cell(t.x, t.y, t.z)

	# Maximum permissible grid coordinates
	nx, ny, nz = box.nx - 1, box.ny - 1, box.nz - 1

	# Include the start point in the hit list
	hits = [ box.cell2cart(p.x, p.y, p.z) ]
	reason = None

	# Keep track of encountered cells for cycle breaks
	hitct = { }

	# Find cell for target points (may be out of bounds)
	ti, tj, tk = <long>t.x, <long>t.y, <long>t.z

	while True:
		# Find the cell for the current test point
		i, j, k = <long>p.x, <long>p.y, <long>p.z

		# Increment and check the cycle counter
		hc = hitct.get((i,j,k), 0) + 1
		if hc > cycles:
			reason = 'cycle'
			break
		hitct[i,j,k] = hc

		if ptdst(p, t) <= 1.0 or (ti == i and tj == j and tk == k):
			# Close enough to destination to make a beeline
			hits.append(box.cell2cart(t.x, t.y, t.z))
			reason = 'destination'
			break

		# Find the function and gradient at the current point
		if not field._evaluate(&fv, &gf, p):
			# Point is out of bounds
			reason = 'boundary'
			break

		# Find the magnitude of the gradient and the search direction
		m = ptnrm(gf)
		pd = scal(-1.0 / m, gf)

		# Establish the baseline Armijo bound
		lim = c * m
		alpha = step

		while alpha >= tol:
			np = axpy(alpha, pd, p)
			# Find the next value
			if field._evaluate(&tv, NULL, np):
				# Stop if Armijo condition is satisfied
				if tv - fv <= -alpha * lim: break
			# Test point out of bounds or failed to satisfy Armijo
			alpha *= tau

		if alpha < tol:
			# Could not find suitable point
			reason = 'stationary'
			break

		# Advance to and record the satisfactory test point
		p = np
		hits.append(box.cell2cart(p.x, p.y, p.z))

	if report: return hits, reason
	else: return hits


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def rytov_zone(p, Box3D box, double l, double s, double width=2, double alpha=pi/4.5):
	'''
	For a linear segment p, either as a Segment3D instance or a sequence
	[start, end] that is compatible with a Numpy array with shape (2, 3),
	identify the Fresnel zone of the segment with wavelength (l * width)
	and, for each cell in the Fresnel zone, compute the value of the Rytov
	sensitivity kernel for a wavelength l using the method outlined in the
	24May18 JPA notes "Rytov Tomographic Reconstruction" assuming a
	wavelength of l. The Rytov kernel is smoothed by a Gaussian of width
	alpha.

	The kernel has a leading coefficient (-2 * pi * s / l**2), where s is
	the inverse of the background sound speed.

	The width and l parameters must be positive.
	'''
	cdef:
		point xt, xr, xp, u, xxt, xxr, ut, ur, dx
		double r, rr, rt, lt, lr, beta, k, fact
		double complex nu, mx, my, mz, delta, apb, phase


	if isinstance(p, Segment3D):
		tup2pt(&xt, p.start)
		tup2pt(&xr, p.end)
	else:
		p = np.asarray(p, dtype='float64')
		if p.shape != (2, 3):
			raise ValueError('Segment p must be a Segment3D or have shape (2, 3)')
		tup2pt(&xt, p[0])
		tup2pt(&xr, p[1])

	if width < 0 or l < 0:
		raise ValueError('Kernel wavelength l and truncation width must be positive')

	k = 2 * pi / l
	fact = s / l
	dx = scal(0.5, box._cell)

	u = axpy(-1, xt, xr)
	r = ptnrm(u)
	iscal(1 / r, &u)

	# Build the Fresnel zone for the path
	path = fresnel_zone(p, box, l * width)

	# Find the Rytov coefficients for each cell in the zone
	rytov = { }
	for (ii, jj, kk) in path:
		tup2pt(&xp, box.cell2cart(ii + 0.5, jj + 0.5, kk + 0.5))
		xxt = axpy(-1, xt, xp)
		xxr = axpy(-1, xr, xp)
		rt = ptnrm(xxt)
		rr = ptnrm(xxr)
		lt = dot(u, xxt)
		if lt < 0: lt = -lt
		lr = dot(u, xxr)
		if lr < 0: lr = -lr

		beta = k / lt + k / lr
		apb = alpha / (alpha + 1j * beta)
		nu = -1j * k * apb

		ut = scal(1 / rt, xxt)
		ur = scal(1 / rr, xxr)

		mx = nu * (ut.x + ur.x)
		my = nu * (ut.y + ur.y)
		mz = nu * (ut.z + ur.z)

		delta = 2 * csinh(mx * dx.x) / mx
		delta *= 2 * csinh(my * dx.y) / my
		delta *= 2 * csinh(mz * dx.z) / mz

		phase = apb * cexp(nu * (rt + rr - r))
		rytov[ii,jj,kk] = fact * (1 / rr + 1 / rt) * cimag(delta * phase)

	return rytov
