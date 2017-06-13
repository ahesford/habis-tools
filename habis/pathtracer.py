'''
Routines for tracing paths through and computing arrival times for slowness
images.
'''

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np

import math
from math import fsum
from numpy.linalg import norm

from itertools import izip

from pycwp.cytools.boxer import Box3D, Segment3D
from pycwp.boxer import Octree
from pycwp.cytools.interpolator import LinearInterpolator3D

from .habiconf import HabisConfigParser, HabisConfigError


class _SurfaceCrossingError(Exception): pass


class SurfaceTracer(object):
	'''
	A class that provides a means for optimizing path integrals through a
	bimodal medium separated by a tesselated surface.
	'''
	def __init__(self, triangles, otree, segmax, edgetol=1e-7):
		'''
		Create a SurfaceTrace object from triangles, a list of N
		pycwp.cytools.boxer.Triangle3D instances. The nodes of all
		triangles in the list should be consistently labeled using a
		global scheme assigning exactly one label to each node in the
		mesh.

		The argument otree must be a pycwp.boxer.Octree instance that
		assigns indices i in [0, len(triangles)) as leaves 
		of an Octree whenever triangles[i].overlaps(box) is True for
		some box in otree. The tree must be built in "multibox" mode,
		assigning each index to all finest-level boxes that overlap the
		corresponding triangle. The tree will be searched to identify
		triangles that intersect paths during optimization.

		Only a reference to otree will be captured; care should be
		taken when modifying the tree after associating it with a
		SurfaceTracer.

		The argument segmax, an integer, specifies a (rough) maximum
		number of segments that will be allowed when defining an
		adaptive optimal path. The maximum is rough because the
		adaptive optimization method may create more segments than
		segmax, but the method will be terminated once the number of
		segments is not less than segmax.

		The argument edgetol, a float, specifies how close
		intersections between a triangle and a linear segment must be
		to the triangle boundary to count as an edge (or vertex)
		intersection rather than an interior hit.
		'''
		# Capture a copy of the triangles list
		self.triangles = tuple(triangles)

		# Capture a reference to the existing Octree
		self.otree = otree

		self.edgetol = float(edgetol)
		self.segmax = int(segmax)

		# Build edge-to-triangle and node-to-triangle maps
		self.ed2tr = { }
		self.nd2tr = { }

		for i, tri in enumerate(self.triangles):
			nodes = tri.labels
			for n0, n1 in izip(nodes, nodes[1:] + nodes[:1]):
				if n0 == n1: raise ValueError('Degenerate triangle')
				ed = min(n0, n1), max(n0, n1)
				try: self.ed2tr[ed].add(i)
				except KeyError: self.ed2tr[ed] = { i }
				if len(self.ed2tr[ed]) > 2:
					raise ValueError('At most two triangles may share a single edge')
				try: self.nd2tr[n0].add(i)
				except KeyError: self.nd2tr[n0] = { i }


	def hitsign(self, path, tri):
		'''
		Return the sign of dot(path, self.triangles[tri]), which is +1
		whenever the inner product is nonnegative and -1 otherwise.
		'''
		triangle = self.triangles[tri]
		# Plane distance is dot(triangle.normal, path) + triangle.offset
		td = triangle.planedist(path)
		return 2 * int(td >= triangle.offset) - 1


	def vthit(self, node, path):
		'''
		For a given node in the global labeling scheme of the mesh in
		self.triangles, find the fractional intersection of a line with
		direction path = (px, py, pz) that passes through the node.

		The method of Linhart (1990) is used, which prescribes that,
		for each triangle tri that contains the node,

		  s(tri) = alpha * sign(dot(path, tri.normal)) / (2 * pi),

		where alpha is the angle subtended by the triangle, projected
		orthogonally to the path, at the node. The overall fractional
		intersection is the sum of s(tri) for all intersected
		triangles.

		Also returned is the average normal at the vertex, with the
		contribution from each intersected triangle weighted by the
		ratio s(tri) / sum(s(tri) for tri in intersections), and a set
		of indices (into self.triangles) of all intersecting triangles.
		'''
		s = 0.0
		normal = np.zeros((3,), dtype=np.float64)
		trset = set(self.nd2tr[node])

		for tri in trset:
			trv = self.triangles[tri]
			# Find the global node in the triangle
			nodes = trv.labels
			try: lnd = nodes.index(n)
			except ValueError: continue
			# Find the angle fraction
			alpha = trv.perpangle(lnd, d)
			st = self.hitsign(path, tri) * alpha / 2 / math.pi
			# Add contribution to average normal
			normal += [st * v for v in trv.normal]
			s += st

		return s, normal / s, trset


	def edhit(self, n0, n1, path):
		'''
		For a given edge with endpoints n0 and n1 in the global
		labeling scheme of the mesh in self.triangles, find the
		fractional intersection of a line with direction path that
		passes through the edge.

		Also returned is the average normal along the edge and a set of
		indices (into self.triangles) of all intersecting triangles.

		If the length of the set is 1, the value of the fractional
		intersection will be +/- 0.5. If the length of the set is 2,
		the value will be 0 or 1.
		'''
		trset = set(self.ed2tr[min(n0, n1), max(n0, n1)])
		# Compute the average normal
		normal = np.sum([self.triangles[tri].normal for tri in trset], axis=0)
		s = sum(0.5 * self.hitsign(path, tri) for tri in trset)
		return s, normal / len(trset), trset


	@staticmethod
	def classify_medium(s, tol=0.2):
		'''
		Classify a medium by the fraction of directional surface
		intersections observed, s:

		* Interior (1): abs(1 - s) < tol,
		* Exterior (0): abs(s) < tol,
		* Indeterminate (-1): otherwise

		The default tolerance is that recommended by Linhart (1990).
		'''
		if abs(1 - tol) < s: return 1
		elif abs(s) < tol: return 0
		else: return -1


	def segcost(self, seg, slowness):
		'''
		Given a segment seg and a pair of values such that slowness[0]
		and slowness[1] are, respectively, exterior and interior
		slowness values, compute and return:

		1. The integral of the slowness over the whole segment,
		2. The gradient of the integral WRT the segment start,
		3. The gradient of the integral WRT to the segment end,
		4. The derivative of the integral WRT slowness[0],
		5. The derivative of the integral WRT slowness[1].

		Interior and exterior portions of seg are determined by
		self.seghits(seg). The method seghits may assign an
		"indeterminate" slowness index of -1 over particular fractions
		of seg. In this implementation, a ValueError will be raised if
		an indeterminate slowness is identified.
		'''
		if len(slowness) != 2:
			raise ValueError('Two slowness values must be provided')

		# Find the surface hits and medium classifications for each segment
		hits, media = self.seghits(seg)

		# The gradients all point in this direction
		l = np.asarray(seg.direction, dtype=np.float64)
		slen = seg.length

		# The total path integral
		pint = 0.0
		# The gradients with respect to endpoints (with an extra work vector)
		gp = np.zeros((3,3), dtype=np.float64)
		# The derivatives with respect to media
		ds = [ 0.0 ] * 2

		# Starting point has no surface normal
		sd = 0
		snrml = np.array([0.] * 3)
		sddn = 0.0

		# Add extra "hit" at end of segment
		for (ed, enrml), med in izip(hits + [(1, np.array([0.] * 3))], media):
			if med not in { 0, 1 }:
				raise ValueError('Values of media must be 0 or 1')
			# Find the slowness for this segment
			slw = slowness[med]
			# Contribution to path integral
			pint += (ed - sd) * slen * slw
			# Contribution to start and end gradients (no surface contributions)
			gp[-1] = (ed - sd) * l
			gp[0] -= slw * gp[-1]
			gp[1] += slw * gp[-1]
			# Gradient surface contribution from left point (if surface exists)
			if not np.allclose(sddn, 0.0):
				gp[-1] = snrml / sddn
				gp[0] += slw * (1 - sd) * gp[-1]
				gp[1] += slw * sd * gp[-1]
			# Gradient surface contribution from right point (if surface exists)
			eddn = np.dot(l, enrml)
			if not np.allclose(eddn, 0.0):
				gp[-1] = enrml / eddn
				gp[0] -= slw * (1 - ed) * gp[-1]
				gp[1] -= slw * ed * gp[-1]
			# Contribution to slowness derivative
			ds[med] += (ed - sd) * slen
			# Cycle end point to start for next round
			sd = ed
			snrml = enrml
			sddn = eddn

		return (pint, gp[0], gp[1], ds[0], ds[1])


	def seghits(self, seg):
		'''
		Identify all intersections between the Segment3D seg and the
		surface defined by self.triangles and classify each consecutive
		sub-segment between the endpoints and points of intersection as
		exterior (0), interior (1), or indeterminate (-1).

		If an intersection falls within a distance self.edgetol of an
		edge or a vertex, all triangles that share the edge or vertex
		will be considered to intersect with the path at the same
		point. Each triangle will intersect with the segment at most
		once. When multiple triangles intersect the path at the same
		point, the intersection will be disregarded if the intersection
		represents a tangency. Otherwise, the intersection will be
		classified as an entry or exit into the volume. Entries, exits
		and tangencies are classified according to the technique
		outlined in Linhart (1990). In some cases, coplanarity of
		(portions of) the path and facets will lead to "indeterminate"
		encounters between the segment and the surface.

		Intersections within a distance of self.edgetol of the endpoint
		of the segment will not be counted as hits to avoid ambiguities
		when the endpoint of a segment coincides with a surface.
		However, these intersections will still be used to identify
		entries and exits in the Linhart algorithm to properly classify
		remaining subsegments.

		Two lists are returned:

		1. hits, a (possibly empty) list of intersection points, each
		   as a pair consisting of the fractional distance along seg
		   and the normal of the surface at this intersection; if the
		   intersection occurs along a vertex or edge, the normal is a
		   weighted average of the normals for all adjoining triangles
		   according to self.vthit or self.edhit as appropriate.

		2. A list of length len(hits) + 1 providing the classifications
		   for sub-segments between consecutive points in the composite
		   list [0] + hits + [1] as interior (1), exterior (0), or
		   indeterminate (-1). The label at index i < len(hits)
		   describes the segment between the points at hits[i - 1] (or,
		   if i == 0, the start of the segment) and hits[i]. The label
		   at index -1 (or len(hits)) describes the sub-segment that
		   ends at the end of the segment and begins at hits[-1] (or,
		   if hits is empty, the beginning of the segment).
		'''
		# Grab the direction of the segment
		sdir = seg.direction

		triangles = self.triangles

		# Use half-line tests for ease in finding interiors
		def bsect(b): return b.intersection(seg, halfline=True)
		def tsect(i):
			# Coplanar encounters can be safely ignored
			try: return triangles[i].intersection(seg, halfline=True)
			except NotImplementedError: return None

		# Find intersections between half-line and surface; cache
		# segment-triangle intersections to avoid redundant tests
		isects = self.otree.search(bsect, tsect, { })

		# Ensure each segment hits each triangle only once
		stris = set()
		hits = [ ]

		for tri, (d, t, u, v) in isects.iteritems():
			# Triangle already counted; only one hit per triangle
			if tri in stris: continue

			# Sort barycentric coordinates to identify boundary hits
			ocrd = sorted(((t, 0), (u, 1), (v, 2)), reverse=True)

			if abs(1 - ocrd[0][0]) <= self.edgetol:
				# This is a vertex hit; find the node
				nd = triangles[tri].labels[ocrd[0][1]]
				# Find the intersection fraction, normal and hit set
				s, nrml, lhits = self.vthit(nd, sdir)
				# Record all triangle hits
				stris.update(lhits)
				# Record the intersection fraction for this point
				hits.append((d, s, nrml))
			elif abs(1 - ocrd[0][0] - ocrd[1][0]) <= self.edgetol:
				# This is an edge hit; find the edge
				n0 = triangles[tri].labels[ocrd[0][1]]
				n1 = triangles[tri].labels[ocrd[1][1]]
				# Find the intersection fraction, normal and hit set
				s, nrml, lhits = self.edhit(n0, n1, sdir)
				# Record all triangle hits
				stris.update(lhits)
				# Record the intersection fraction for this point
				hits.append((d, s, nrml))
			else:
				# Record a single-triangle hit
				stris.add(tri)
				nrml = np.array(triangles[tri].normal)
				# Record intersection fraction for this point
				hits.append((d, self.hitsign(sdir, tri), nrml))

		# Sort the intersections by length along the segment
		hits.sort()

		# Find midpoints of segments
		midpoints = [ ]
		ihits = [ ]

		lastd = 0
		for d, s, nrm in hits:
			# Ignore hits close to or beyond segement endpoints
			if d > 1 - self.edgetol: break
			elif d < self.edgetol: continue

			# Find the midpoint of the sub-segment
			midpoints.append(0.5 * (d + lastd))
			# Add the intersection and normal to hit list
			ihits.append((d, nrm))
			lastd = d

		# Add the midpoint for the end segment with an implicit 1
		midpoints.append(0.5 * (1 + lastd))

		# Keep track of the interior/exterior classification
		intext = [ ]

		# Count intersections from the far end
		st = 0.0

		try:
			for d, s, nrm in reversed(hits):
				while d < midpoints[-1]:
					# Crossed a midpoint; classify medium
					intext.append(self.classify_medium(st))
					# Strip the just-counted midpoint
					midpoints.pop()
					# Nothing left if all midpoints classified
					if not midpoints: raise _SurfaceCrossingError
				# Add next interface to intersection count
				st += s
		except _SurfaceCrossingError:
			pass
		else:
			# All hits exhausted without catching the last midpoint
			if len(midpoints) != 1:
				raise ValueError('Expected one midpoint after counting all hits')
			intext.append(self.classify_medium(st))

		# Media were counted in reverse order
		intext.reverse()

		if len(intext) != len(ihits) + 1:
			raise ValueError('Expected number of segments to exceed number of hits by 1')

		return ihits, intext


	def pathcost(self, points, slowness):
		'''
		Given an N-by-3 list points that defines a piecewise linear
		path and a pair of values such that slowness[0] and slowness[1]
		are, respectively, exterior and interior slowness values,
		compute and return:

		1. The integral of the slowness over the path,
		2. The gradient of the integral WRT interior points, and
		3. The derivative of the integral WRT each slowness value.

		The point gradients, slowness derivatives and integrals for
		each linear segment are computed with self.segcost.

		By convention, the ends (points[0] and points[-1]) of the path
		are considered fixed and the gradients of the path integral
		with respect to these points is always 0.
		'''
		points = np.asarray(points, dtype=np.float64)
		n, d = points.shape
		if d != 3 or n < 2:
			raise ValueError('Argument "points" must have shape (N,3) for some N > 1')

		pint = 0
		pgrad = np.zeros_like(points)
		psdiff = [ 0.0 ] * 2

		for i, (l, r) in enumerate(izip(points, points[1:])):
			pi, pgl, pgr, ps1, ps2 = self.segcost(Segment3D(l,r), slowness)
			pint += pi
			pgrad[i] += pgl
			pgrad[i+1] += pgr
			psdiff[0] += ps1
			psdiff[1] += ps2

		# Kill gradients at endpoints
		pgrad[0] = 0.
		pgrad[-1] = 0.

		return pint, pgrad, psdiff


	def pathhits(self, points):
		'''
		Given an N-by-3 array points that defines a piecewise linear
		path, return as an M-by-3 array, with M >= N, the combined list
		of control points and intersections between the path and the
		surface described by self.triangles. The points are ordered
		naturally, by increasing distance along the path.

		The method self.seghits is used to identify hits for each path
		segment. Because self.seghits skips hits near segment
		endpoints, a single hit will be counted for each instance where
		the final medium of one segment differs from the initial medium
		for the next segment. This extra hit will be coincident with
		the common control point between each segment.
		'''
		phits = [ points[0] ]

		for l, r in izip(points, points[1:]):
			# Find intersections along the segment
			seg = Segment3D(l, r)
			hits, media = self.seghits(seg)
			phits.extend(seg.cartesian(d) for d, _ in hits)
			# Add the right to the hit list
			phits.append(r)

		return np.asarray(phits, dtype=np.float64)


	def trace(self, slowness, src, rcv, bfgs_opts={}):
		'''
		Given a bimodal slowness = (s0, s1), where s0 is the slowness
		on the exterior of the surface defined by the mesh represented
		in self.triangles and s1 is the slowness on the interior of the
		surface, trace an optimum (minimum-time) path from a source
		with world coordinates src = (sx, sy, sz) to a receiver with
		world coordinates rcv = (rx, ry, rz).

		If the path does not intersect the surface, a single linear
		segment from source to receiver will be returned. Otherwise,
		the path will be an N-by-3 array of control points for a
		piecewise linear path from source to receiver.

		The path tracing is adaptive; control points will be added and
		optimization repeated until an optimal path is found that
		intersects the surface exactly (N - 2) times for an N-point
		path.

		Alongside the returned path, the value of the slowness integral
		will also be returned.

		The dictionary bfgs_opts will be passed as keyword arguments to
		scipy.optimize.fmin_l_bfgs_b to optimize each path.
		'''
		from scipy.optimize import fmin_l_bfgs_b

		points = np.array([src, rcv], dtype=np.float64)
		if points.shape != (2, 3):
			raise ValueError('Both src and rcv must be 3-D coordinates')

		# Build an optimizer for fmin_l_bfgs_b
		def ffg(x):
			# Reshape to a simple points list
			pts = x.reshape((-1, 3), order='C')
			pint, pgrad, _ = self.pathcost(pts, slowness)
			# Flatten the gradient
			return pint, pgrad.ravel('C')

		# Compute the straight-through path integral
		pathint = self.pathcost(points, slowness)[0]

		while len(points) < self.segmax + 1:
			# Trace the path to find intersections
			npoints = self.pathhits(points)
			# No new points were found, stop trying to optimize
			if len(npoints) <= len(points): break
			# Optimize the new path
			npoints, pathint, info = fmin_l_bfgs_b(ffg, npoints, **bfgs_opts)
			# Reshape the optimized path
			points = npoints.reshape((-1, 3), order='C')

		# Return the fixed path and its path integral
		return points, pathint


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


	def trace(self, si, src, rcv, intonly=False):
		'''
		Given an interpolatored slowness map si (as a 3-D Numpy array
		or pycwp.cytools.interpolator.Interpolator3D), a source with
		world coordinates src and a receiver with world coordinates
		rcv, trace an optimum path from src to rcv using

		  si.minpath(gs, gr, self.nmax, self.itol,
		  		self.ptol, self.box.cell, **self.optargs),

		where gs and gr are grid coordinates of src and rcv,
		respectively, according to self.box.cart2cell.

		If si is a Numpy array, it will be converted to an instance of
		(a descendant of) Interpolator3D as

		  si = pycwp.cytools.interpolator.LinearInterpolator3D(si).

		If intonly is True, only the integral of the slowness over the
		optimum path will be returned. Otherwise, the optimum path will
		be marched through self.box to produce a map (i, j, k) -> L,
		where (i, j, k) is a cell index in self.box and L is the
		accumulated length of the optimum path through that cell. The
		return value in this case will be (pathmap, pathint), where
		pathmap is this cell-to-length map and pathint is the
		integrated slowness over the optimum path.

		Any Exceptions raised by si.minpath will not be caught by this
		method.
		'''
		box = self.box

		try: shape = si.shape
		except AttributeError:
			# Treat a non-structured si as a Numpy array
			si = np.asarray(si)
			shape = si.shape

		if shape != box.ncell:
			raise ValueError('Shape of si must be %s' % (box.ncell,))

		# Convert world coordinates to grid coordinates
		gsrc = box.cart2cell(*src)
		grcv = box.cart2cell(*rcv)

		# Make sure slowness is an interpolator
		if not hasattr(si, 'minpath'): si = LinearInterpolator3D(si)

		# Use preconfigured options to evaluate minimum path
		popt, pint = si.minpath(gsrc, grcv, self.nmax,
				self.itol, self.ptol, box.cell, **self.optargs)

		# If only the integral is desired, just return it
		if intonly: return pint

		# Convert path to world coordinates and march
		points = np.array([box.cell2cart(*p) for p in popt])
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
