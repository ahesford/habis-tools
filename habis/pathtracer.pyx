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

from itertools import izip

from pycwp.cytools.boxer import Box3D

from .habiconf import HabisConfigParser, HabisConfigError

from libc.stdlib cimport rand, RAND_MAX

from pycwp.cytools.ptutils cimport *
from pycwp.cytools.interpolator cimport Interpolator3D
from pycwp.cytools.quadrature cimport Integrable

cdef inline double randf() nogil:
	'''
	Return a sample of a uniform random variable in the range [0, 1].
	'''
	return <double>rand() / <double>RAND_MAX

ctypedef struct PathIntContext:
	point a
	point b
	bint dograd

class TraceError(Exception): pass


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
	def minpath(self, start, end, unsigned long nmax, double itol, double ptol,
			h=1.0, double perturb=0.0, unsigned long nstart=1,
			bint warn_on_fail=True, bint raise_on_fail=False, **kwargs):
		'''
		Given 3-vectors start and end in grid coordinates, search for a
		path between start and end that minimizes the path integral of
		the function interpolated by self.data.

		The path will be iteratively divided into at most N segments,
		where N = 2**M * nstart for the smallest integer M that is not
		less than nmax. With each iteration, an optimal path is sought
		by minimizing the object self.pathint(path, itol, h) with
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
		bf = lf = self.pathint(points, itol, h, False)

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
			xopt, nf, info = bfgs(self.pathint, points,
					fprime=None, args=(itol, h, True), **kwargs)
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
	def pathint(self, points, double tol, h=1.0, bint grad=False):
		'''
		Given control points specified as rows of an N-by-3 array of
		grid coordinates, use an adaptive Simpson's rule to integrate
		the image associated with self.data along the piecewise linear
		path between the points.

		As a convenience, points may also be a 1-D array of length 3N
		that represents the two-dimensional array of points flattened
		in C order.

		Each segment of the path will be recursively subdivided until
		integration converges to within tol or the recursion depth
		exceeds a limit that ensures that step sizes do not fall
		below machine precision.

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
		# Evaluate the integrand at the left endpoint
		if not self.integrand(ends, 0., <void *>(&ctx)):
			raise ValueError('Cannot evaluate integrand at point %s' % (pt2tup(ctx.a),))

		for i in range(1, npts):
			# Initialize the right point
			ctx.b = packpt(pts[i,0], pts[i,1], pts[i,2])
			if not self.integrand(&(ends[nval]), 1., <void *>&(ctx)):
				raise ValueError('Cannot evaluate integrand at point %s' % (pt2tup(ctx.b),))
			# Calculate integrals over the segment
			if not self.simpson(results, nval, tol, <void *>(&ctx), ends):
				raise ValueError('Cannot evaluate integral from %s -> %s' % (pt2tup(ctx.a), pt2tup(ctx.b)))

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
	cdef bint integrand(self, double *results, double u, void *ctx) nogil:
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
		if sctx == <PathIntContext *>NULL: return False

		# Find the evaluation point
		p = lintp(u, sctx.a, sctx.b)

		if sctx.dograd:
			if not self.data._evaluate(&fv, &gf, p): return False
			mu = 1 - u
			results[0] = fv
			pt2arr(&(results[1]), scal(1 - u, gf))
			pt2arr(&(results[4]), scal(u, gf))
		else:
			if not self.data._evaluate(&fv, <point *>NULL, p): return False
			results[0] = fv
		return True


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

		try: itol = config.get(psec, 'itol', mapper=float, default=1e-5)
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
		self.itol = float(itol)
		self.nmax = int(segmax)

		# Copy the optimization arguments
		self.optargs = dict(optargs)


	def trace(self, si, src, rcv, fresnel=0, intonly=False):
		'''
		Given an interpolatored slowness map si (as a 3-D Numpy array
		or pycwp.cytools.interpolator.Interpolator3D), a source with
		world coordinates src and a receiver with world coordinates
		rcv, trace an optimum path from src to rcv using

		  PathIntegrator(si).minpath(gs, gr, self.nmax, self.itol,
		  		self.ptol, self.box.cell, **self.optargs),

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
		popt, pint = integrator.minpath(gsrc, grcv,
				self.nmax, self.itol, self.ptol,
				box.cell, raise_on_fail=True, **self.optargs)

		# If only the integral is desired, just return it
		if intonly and fresnel <= 0: return pint

		# Convert path to world coordinates for marching
		points = np.array([box.cell2cart(*p) for p in popt])

		if fresnel > 0:
			# Trace the Fresnel zone through the box
			plens = box.fresnel(points, fresnel)

			# Compute the total length of the path
			tlen = fsum(norm(r - l) for r, l in izip(points, points[1:]))

			# Convert Fresnel-zone weights to integral contributions
			wtot = fsum(plens.itervalues())
			plens = { k: tlen * v / wtot for k, v in plens.iteritems() }

			# Reintegrate over Fresnel path
			pint = fsum(v * si.evaluate(*k, grad=False)
					for k, v in plens.iteritems())

			return pint if intonly else (plens, pint)

		marches = box.raymarcher(points)

		# Make sure single-segment march still a list
		if points.shape[0] < 3: marches = [marches]
		# Accumulate the length of each path in each cell
		plens = { }
		for (st, ed), march in izip(izip(points, points[1:]), marches):
			# Compute whol length of this path segment
			dl = norm(ed - st)
			for cell, (tmin, tmax) in march.iteritems():
				# Convert fractional length to real length
				# 0 <= tmin <= tmax <= 1 guaranteed by march algorithm
				contrib = (tmax - tmin) * dl

				# Add the contribution to the list for this cell
				try: plens[cell].append(contrib)
				except KeyError: plens[cell] = [contrib]

		# Accumulate the contributions to each cell
		# Return the path map and the path-length integral
		return { k: fsum(v) for k, v in plens.iteritems() }, pint


def srcompensate(trpairs, elements, bx, si, N):
	'''
	For a given list of (t, r) pairs trpairs that describe transmit-receive
	indices into elements, a coordinate grid defined by the Box3D bx, an
	interpolated slowness image si (as an Interpolator3D instance) defined
	on the grid, and a desired quadrature order N, return a list of tuples
	(I, dl, rn) for each entry in trpairs, where I is the compensated
	straight-ray arrival time, dl is the unit direction of the straight-ray
	path from elements[t] to elements[r], and rn is the final wavefront
	normal at the receiver.
	'''
	# Verify grid
	cell = np.array(bx.cell)
	if bx.ncell != si.shape:
		raise ValueError('Grid of bx and si must match')

	# Build the quadrature weights
	from pycwp.cytools.quadrature import glpair
	glwts = [ glpair(N, i) for i in xrange(N) ]

	# Trace the paths one-by-one
	results = [ ]
	for t, r in trpairs:
		ival = 0.
		# Find element coordinates and convert to grid coordinates
		tx = np.array(elements[t])
		rx = np.array(elements[r])
		tg = np.array(bx.cart2cell(*tx))
		rg = np.array(bx.cart2cell(*rx))

		# Find unit direction and length of path
		dl = rx - tx
		L = norm(dl)
		dl /= L

		# Initial wavefront normal is direction vector
		lu = 0.
		ln = dl

		# First evaluation point is transmitter
		x = tg
		sv, svg = si.evaluate(*x)
		# Scale gradient properly
		svg = (cell * svg) / sv
		# Find wavefront "drift"
		ndl = np.dot(ln, dl)

		for gl in glwts:
			# Next evaluation point
			ru = 0.5 * (gl.x() + 1.)
			# New wavefront normal (renormalized) and "drift"
			ln = ln + (ru - lu) * L * svg / ndl
			ln /= norm(ln)
			ndl = np.dot(ln, dl)
			# New evaluation point and function evaluation
			x = (1. - ru) * tg + ru * rg
			sv, svg = si.evaluate(*x)
			svg = (cell * svg) / sv
			# Update total integral
			ival += 0.5 * gl.weight * ndl * sv
			lu = ru
		results.append((L * ival, dl, ln))
	return results
