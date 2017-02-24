'''
Routines for tracing paths through and computing arrival times for slowness
images.
'''

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np

from math import fsum
from numpy.linalg import norm

from itertools import izip

from pycwp.cytools.boxer import Box3D
from pycwp.cytools.interpolator import LinearInterpolator3D

from .habiconf import HabisConfigParser, HabisConfigError

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
		self.box = Box3D(lo, hi)
		self.box.ncell = ncell

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
