#!/usr/bin/env python

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

'''
Perform an arrival-time-tomographic udpate of sound speeds.
'''

import sys, os

import numpy as np
from numpy.linalg import norm

from scipy.optimize import fmin_l_bfgs_b
from scipy.ndimage import median_filter

from math import fsum

from random import sample

from time import time

from itertools import izip

from mpi4py import MPI

from pycwp.cytools.boxer import HermiteInterpolator3D, LinearInterpolator3D, Box3D

from habis.habiconf import HabisConfigParser, HabisConfigError, matchfiles
from habis.formats import loadkeymat as ldkmat, loadmatlist as ldmats
from habis.mpdfile import flocshare


def getatimes(atfile, column=0, start=0, stride=1):
	'''
	Read the 2-key arrival-time map with name atfile and remove
	backscatter entries for which key[0] == key[1]. The map is always
	treated as a multi-value map (i.e., 'scalar' is False in loadkeymat).
	The column argument specifies which value index should be retained.

	Only every stride-th *valid* record, starting with the start-th such
	record, is retained.
	'''
	if not 0 <= start < stride:
		raise ValueError('Index start must be at least zero and less than stride')

	# Load the map, eliminate invalid elemenets, and keep the right portion
	atimes = { }
	idx = 0

	for (t, r), v in ldkmat(atfile, nkeys=2, scalar=False).iteritems():
		# Ignore backscatter
		if t == r: continue
		# Keep every stride-th valid record
		if idx % stride == start: atimes[t,r] = v[column]
		# Increment the valid record count
		idx += 1

	return atimes


def traceloop(box, elements, atimes, dl=1e-3, hmax=0.25,
		segmax=256, pathtol=1e-3, slowdef=None, bfgs_opts={},
		interp=LinearInterpolator3D, comm=MPI.COMM_WORLD):
	'''
	Establish an MPI worker loop that performs path tracing for a list of
	source-receive pairs (whose coordinates will be pulled from the map
	elements) through the given Box3D box. Path tracing is done by
	perturbing the straight-ray path from source to receiver using L-BFGS
	to minimize the travel-time integral through a slowness field. The
	bfgs_opts dictionary can be used to provide keyword arguments to
	configure L-BFGS (actually scipy.optimize.fmin_l_bfgs_b).

	The trace results, which map each transmit-receive pair to a map from
	cell indices (in the Box3D) to lengths of the (t, r) path through that
	cell, are used to compute a process-local share of the cost function

		C(s) = 0.5 * || Ls - D ||**2,

	where L is a representation of the path-length operator mapping the
	slowness at each cell in s to its contribution to the travel times
	based on the traced paths, and the data D consists of corresponding
	entries atimes[t,r]. The worker also evaluates the gradient of this
	cost function, which is given by

		grad(C)(s) = L^T [ Ls - D ].

	By default, the local share of the cost function and its gradient
	encompass the full list of transmit-receive pairs in the arrival-time
	map atimes. Thus, each process running a traceloop should receive a
	distinct arrival-time map.

	All communication will happen over the MPI communicator comm, with the
	root rank (0) representing the master node that is not engaged in this
	loop. Flow control is achieved by calling, from the master, one of

		comm.bcast("SLOWNESS"),
		comm.bcast("SCRAMBLE,<N>"),
		comm.bcast("EVALUATE"), or
		comm.bcast("ENDTRACE").

	Note that these flow-control messages use the higher-level Python
	object interface.

	Workers must be provided a slowness map, s, before evaluating the cost
	function and gradient. This is accomplished by first broadcasting a
	"SLOWNESS" control message to the workers. After the control message,
	a Numpy array s, with shape box.ncell and dtype np.float64, must be
	broadcast to the workers by calling comm.Bcast(s), using the low-level
	buffer interface, on the root. The slowness map will be retained for
	all future evaluations, but the map can be replaced at any time with a
	subsequent transmission of the SLOWNESS control message and a new
	slowness array.

	For optimization, the slowness is interpolated by calling interp(s) to
	obtain an instance of pycwp.cytools.boxer.Interpolator3D. If the
	slowdef argument is not None, the value of slowdef will be assigned to
	the "default" property of the interpolator returned by interp(s) to
	provide a default (out-of-bounds) value. Otherwise, any path that
	reaches an out-of-bounds slowness will be excluded from the cost
	function.

	To probabilistically restrict the number of transmit-receive pairs
	employed in the cost function, a "SCRAMBLE,<N>" control message may be
	broadcast from the root, where "<N>" is a string representation of a
	positive integer. When this message is received by the workers, each
	worker will randomly select N transmit-receive pairs from atimes to be
	used in all future evaluations of the cost function. The SCRAMBLE
	message may be sent multiple times, with a new random selection being
	made after each message is received. If N is less than 1 or no less
	than the length of atimes, all transmit-receive pairs will be used.

	To evaluate the cost function and its gradient, broadcast an "EVALUATE"
	message to the workers from the root. Upon receipt of the message, each
	worker will evaluate its share of the global cost function and
	gradient. Because the local contributions are subdivided along the
	range (transmit-receive pairs) of the path-length operator L, the
	global cost and gradient can be obtained by summing local values from
	each node.

	After broadcasting an "EVALUATE" control message, the root must first
	accumulate the scalar cost functions by calling the high-level Python
	object method comm.reduce twice in succession. The first call will
	accumulate the values of the cost functions. The second call will
	accumulate the number of participating rows in the path-length operator.

	After accumulating the scalar cost and row count, the root must
	accumulate the global gradient by calling the low-level buffer method
	comm.Reduce(input, output, op=MPI.SUM). The accumulated gradient will
	be stored in output, which should be an empty Numpy array with shape
	box.ncell and dtype np.float64. The input array should have the same
	shape and data type, and will be added to the global gradient. To
	capture the pure gradient alone, zero the output prior to the reduction
	and call comm.Reduce(MPI.IN_PLACE, output, op=MPI.SUM).

	Upon receipt of an "ENDTRACE" control message, the worker loop will
	terminate.
	'''
	rank, size = comm.rank, comm.size

	# Note that the slowness and its path optimizer
	s = None
	popt = None

	# Make sure all arrival-time entries have matching entries in elements
	atimes = { k: v for k, v in atimes.iteritems()
			if k[0] in elements and k[1] in elements }

	# By default, work with all of the arrival times
	trset = sorted(atimes.iterkeys())

	# Keep storage for the gradient and corrections to the running sum
	gf = np.zeros(box.ncell, dtype=np.float64, order='C')

	# Work loop
	while True:
		# Receive message
		msg = comm.bcast(None).strip().upper()

		# Terminate the loop
		if msg == 'ENDTRACE':
			break
		elif msg == 'SLOWNESS':
			# Allocate new storage for the received slowness
			if s is None: s = np.empty(box.ncell, dtype=np.float64)
			# Receive the new slowness in pre-existing storage
			comm.Bcast(s)
			# Build the interpolator
			si = interp(s)
			if slowdef is not None: si.default = slowdef
			# Build the path optimizer
			popt = makeoptimizer(si, dl, hmax)
			continue
		elif msg.startswith('SCRAMBLE,'):
			try: ntr = int(msg.split(',')[1])
			except Exception as e:
				print 'MPI rank %d of %d: bad SCRAMBLE message:', str(e)
				comm.Abort(1)
			if 0 < ntr < len(atimes):
				trset = sorted(sample(atimes.keys(), ntr))
			else: trset = sorted(atimes.iterkeys())
			continue
		elif msg != 'EVALUATE':
			print 'MPI rank %d of %d: invalid MPI tag %d' % (rank, size, tag)
			comm.Abort(1)

		if popt is None:
			print 'MPI rank %d of %d: define slowness before evaluation' % (rank, size)
			comm.Abort(1)

		# Accumulate the local cost function and gradient
		f = []
		gf[:,:,:] = 0.0
		nrows, nskip = 0L, 0L

		# Compute contributions for each source-receiver pair
		for t, r in trset:
			plens = pathtrace(box, popt, elements[t],
					elements[r], segmax, pathtol, bfgs_opts)
			if not plens:
				nskip += 1
				continue
			nrows += 1
			# Calculate error in model arrival time
			err = fsum(s[c] * l for c, l in plens.iteritems()) - atimes[t,r]
			f.append(err**2)
			# Add gradient contribution with Kahan summation
			for c, l in plens.iteritems(): gf[c] += l * err

		if nskip: print 'MPI rank %d of %d: skipped %d untraceable paths' % (rank, size, nskip)

		# Transmit the scalar cost function and row count
		comm.reduce(0.5 * fsum(f), op=MPI.SUM)
		comm.reduce(nrows, op=MPI.SUM)
		# Use the lower-level routine for the arrays
		comm.Reduce(gf, None, op=MPI.SUM)


def pathinterp(paths):
	'''
	Given a piecewise linear curve defined by segments between successive
	points in path, return an interpolated path that inserts a control
	point at the midpoint of each segment.

	Returns a new list ordered in the same fashion as paths.
	'''
	# Make sure the path is an array of points
	paths = np.asarray(paths)
	if paths.ndim != 2:
		raise ValueError('Path array must be 2-D')

	# Allocate storage for the interpolated path
	nseg = len(paths) - 1
	nnpt = 2 * nseg + 1
	npaths = np.empty((nnpt, paths.shape[1]), dtype=paths.dtype)

	# Copy starting point
	npaths[0] = paths[0]

	# Add new midpoints and existing endpoints
	for i in xrange(1, len(paths)):
		i2 = 2 * i
		npaths[i2 - 1] = 0.5 * (paths[i - 1] + paths[i])
		npaths[i2] = paths[i]

	return npaths


def makeoptimizer(si, dl, h):
	'''
	Based on si, build an optimization function that takes two arguments:

	* x, an (N, 3) array, flattened in C order, that represents N control
	  points of a test path through the interpolated slowness field si.

	* costonly, an optional Boolean which is False by default. When the
	  argument is True, the optimization function returns only

	  	si.pathint(xr, h)

	  When the argument is False, the optimization returns the tuple

	  	si.pathint(xr, h), si.pathgrad(xr, dl, h).ravel('C'),

	  where xr = x.reshape((-1, 3), order='C').

	The function is meant for use in fmin_l_bfgs_b when costonly is False.
	'''
	def ffg(x, costonly=False):
		x = x.reshape((-1, 3), order='C')
		f = si.pathint(x, h)
		if costonly: return f
		g = si.pathgrad(x, dl, h)
		return f, g.ravel('C')

	return ffg


def pathtrace(box, optimizer, src, rcv, nmax, tol=1e-3, bfgs_opts={}):
	'''
	Given a source-receiver pair with Cartesian coordinates src and rcv,
	respectively, and a Box3D box that defines an image grid, minimize the
	optimizer (the output of makeoptimizer) over a path from src to rcv
	using L-BFGS. The minimization is done adaptively, iteratively
	subdividing the segment between src and receive into 2**i segments at
	iteration i and finding an optimal solution for the subdivided path.
	Iteration stops when the optimizer fails to improve its value by more
	than tol (in a relative sense), or when the number of path segments
	meets or exceeds nmax.

	The trace will abort (by returning an empty path) if the optimized
	function for a path in one iteration is larger than the optimized value
	in the preceding iteration.

	The determined path will be marched through the box to produce a
	map from cell indices in the box to the length of the intersection of
	that cell and the path (i.e., the map is a path-length matrix for a
	single path). The resulting path-length map will be returned.

	In the case of any failure that prevents the determination of a
	convergent, optimized path-length map, an empty map will be returned.
	'''
	# Start from a straight-ray assumption in low contrast
	points = np.array([box.cart2cell(*src), box.cart2cell(*rcv)])
	pbest = points

	# Start counting interations
	nit = 0
	# Find the cost for the first step
	lf = optimizer(points, costonly=True)
	bf = lf

	while len(points) < nmax:
		# Double the number of segments
		points = pathinterp(points)

		# Optimize control points; on failure, abandon path
		try: xopt, nf, inf = fmin_l_bfgs_b(optimizer, points, **bfgs_opts)
		except ValueError as e: return { }

		points = xopt.reshape((-1, 3), order='C')

		if nf < bf:
			bf = nf
			pbest = points

		# Check convergence
		if abs(nf - lf) < tol * abs(nf): break
		lf = nf

	# Convert path to Cartesian coordinates and march segments
	points = np.array([box.cell2cart(*p) for p in pbest])
	try: marches = box.raymarcher(points)
	except ValueError as e:
		print 'NOTE: ray march failed:', (src, rcv), str(e)
		return { }

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

	# Safely accumulate the contributions to each cell
	return { k: fsum(v) for k, v in plens.iteritems() }


def makeimage(s, mask, box, nm, nrounds, bounds=None, mfilter=3,
		bfgs_opts={}, partial_output=None, comm=MPI.COMM_WORLD):
	'''
	Iteratively compute a slowness image by updating the given profile s
	defined over the grid defined in the Box3D box. If mask is not None, it
	should be a bool array with the same shape as s (and box.ncell) that is
	True for cells that should be updated and False for cells that should
	remain unchanged.

	On each rank, nm measurements will be randomly selected from the pool
	of available measurements to perform each of a total of nrounds
	reconstruction rounds. After each round of randomized reconstruction,
	if mfilter has a Boolean value of True, a median filter of size mfilter
	will be applied to the image before beginning the next round. The
	argument mfilter can be a scalar or a three-element sequence of
	positive integers.

	The image is formed using L-BFGS-B to minimize the global accumulation
	of cost functions as computed by workers running traceloop(). If bounds
	is not None, it should be a tuple of the form (sl, sh), for which a
	per-pixel list of slowness bounds will be computed so that all values
	in the final image fall in the range [sl, sh]. The bfgs_opts dictionary
	will be passed as keyword arguments to scipy.optimize.fmin_l_bfgs_b.

	If partial_output is not None, it should be a string specifying a name
	template that will be rendered to store images produced after each
	update. An "update" counts as a single iteration of L-BFGS-B within a
	single round of randomized reconstruction. The final output name will
	be produced by calling partial_output.format(round=round, iter=iter),
	where "round" and "iter" are the randomized round and inner L-BFGS-B
	iteration numbers, respectively. If partial_output is None, no partial
	images will be stored.

	Participating ranks will be pulled from the given MPI communicator,
	wherein rank 0 must invoke this function and all other ranks must
	invoke traceloop().
	'''
	# Make sure the image is the right shape
	s = np.array(s, dtype=np.float64)

	if mask is None: mask = np.ones(s.shape, dtype=bool)
	else: mask = np.asarray(mask, dtype=bool)

	if s.shape != box.ncell or s.shape != mask.shape:
		raise ValueError('Shape of s and optional mask must be %s' % (box.ncell,))

	nnz = np.sum(mask.astype(np.int64))

	# Work arrays
	sp = np.empty(s.shape, dtype=np.float64)
	gf = np.empty(s.shape, dtype=np.float64)
	x = np.empty((nnz,), dtype=np.float64)

	def ffg(x):
		'''
		This function returns the cost functional and its gradient for
		optimization by fmin_l_bfgs_b to obtain a contrast update.

		The cost function is C(s + x) + 0.5 * norm(x)**2 and its
		gradient is grad(C)(s) + x, were C and grad(C) are as described
		in the traceloop docstring.
		'''
		# Track the run time
		stime = time()

		# Reshape slowness
		sp[:,:,:] = s
		sp[mask] += x

		# Send the slowness to the worker pool
		comm.bcast("SLOWNESS")
		comm.Bcast(sp)

		# The time to transmit sound-speed map
		txtime = time()

		# Reset path traces and perform cost-function and gradient evaluation
		comm.bcast("EVALUATE")

		# Accumulate cost and row counts
		f = comm.reduce(0.0, op=MPI.SUM)
		nmeas = comm.reduce(0L, op=MPI.SUM)
		f /= nmeas

		# Accumulate gradient and extract changeable portion
		gf[:,:,:] = 0
		comm.Reduce(MPI.IN_PLACE, gf)
		lgf = gf[mask] / nmeas

		# The time to evaluate the function and gradient
		etime = time() - txtime
		txtime -= stime

		print 'Cost evaluation times (sec): %0.2f bcast, %0.2f eval' % (txtime, etime)

		return f, lgf

	# Make sure any bounds are ordered properly
	if bounds is not None:
		sl, sh = bounds
		if sh < sl: bounds = (sh, sl)
		else: bounds = (sl, sh)

	# Iteratively select random transmit-receive pairs to update an image
	for i in range(nrounds):
		if bounds:
			# Clip slowness on start
			s[:,:,:] = np.clip(s, bounds[0], bounds[1])
			# Develop per-sample bounds
			vxbounds = [(bounds[0] - sv, bounds[1] - sv) for sv in s[mask]]
		else:
			vxbounds = None

		# Clear the initial perturbation guess
		x[:] = 0

		# Scramble the set of arrival times
		comm.bcast("SCRAMBLE,%d" % nm)

		# Build a callback to write per-iteration results, if desired
		cb = callbackgen(partial_output, s, mask, i)

		# Perform the iterative update a limited number of times
		x, f, info = fmin_l_bfgs_b(ffg, x,
				bounds=vxbounds, callback=cb, **bfgs_opts)

		# Update the image
		s[mask] += x
		# Apply a desired filter
		if mfilter: s[:,:,:] = median_filter(s, size=mfilter)

		print 'Round', i, 'complete, residual', f

	return s


def callbackgen(templ, s, mask, rnd):
	'''
	Build a callback to store partial images of perturbations to the
	assumed slowness s, with the given update mask (True where samples will
	be perturbed, False elsewhere), for a given randomized round rnd.

	The callback will track total iterations within the round and will
	store images in npy format with the name

		templ.format(round=rnd, iter=i),

	where i is an internally incremented counter.
	'''
	if not templ: return None

	nit = [0]
	def callback(x):
		# Write out the iterate
		sp = s.copy()
		sp[mask] += x
		fname = templ.format(round=rnd, iter=nit[0])
		np.save(fname, sp.astype(np.float32))
		nit[0] += 1

	return callback


def usage(progname=None, retcode=1):
	if not progname: progname = os.path.basename(sys.argv[0])
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname
	sys.exit(int(retcode))


if __name__ == "__main__":
	if len(sys.argv) != 2: usage()

	try:
		config = HabisConfigParser(sys.argv[1])
	except Exception as e:
		err = 'Unable to load configuration file %s' % sys.argv[1]
		raise HabisConfigError.fromException(err, e)

	tsec = 'tomogrid'

	def _throw(msg, e, sec=None):
		if not sec: sec = tsec
		raise HabisConfigError.fromException(msg + ' in [%s]' % (sec,), e)

	try:
		# Load the image grid
		lo = config.get(tsec, 'lo')
		hi = config.get(tsec, 'hi')
		ncell = config.get(tsec, 'ncell')
		bx = Box3D(lo, hi)
		bx.ncell = ncell
	except Exception as e:
		_throw('Configuration must specify valid grid', e, 'tomogrid')

	rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

	tsec = 'tomomaster' if not rank else 'tomoslave'

	# Read optional BFGS option dictionary
	# Master uses this for image updates, slaves for path tracing
	try:
		bfgs_opts = config.get(tsec, 'optimizer',
				mapper=dict, checkmap=False, default={ })
	except Exception as e: _throw('Invalid optional optimizer', e)

	# Make a communicator for just the workers
	wcolor = MPI.UNDEFINED if not rank else 1
	wcomm = MPI.COMM_WORLD.Split(wcolor, rank)

	if not rank:
		print 'Solution defined on grid', bx.lo, bx.hi, bx.ncell

		try:
			# Find the slowness map
			s = config.get(tsec, 'slowness')
			# Load the map or build a constant map
			try: s = float(s)
			except (ValueError, TypeError): s = np.load(s).astype(np.float64)
			else: s = s * np.ones(bx.ncell, dtype=np.float64)
		except Exception as e: _throw('Configuration must specify slowness', e)


		# Create guess and load solution map
		try:
			mask = config.get(tsec, 'slowmask', default=None)
			if mask: mask = np.load(mask).astype(bool)
		except Exception as e: _throw('Invalid optional slowmask', e)

		try:
			# Read the number of measurements per round
			nmeas = config.get(tsec, 'nmeas', mapper=int, default=0)
			if nmeas < 0: raise ValueError('nmeas must be nonnegative')
		except Exception as e: _throw('Invalid optional nmeas', e)

		try:
			# Read the number of rounds
			nrounds = config.get(tsec, 'rounds', mapper=int, default=1)
			if nrounds < 1: raise ValueError('rounds must be positive')
		except Exception as e: _throw('Invalid optional rounds', e)

		# Read optional the filter parameters
		mfilter = config.get(tsec, 'mfilter', default=None)
		# Read optional partial_output format string
		partial_output = config.get(tsec, 'partial_output', default=None)

		# Read required final output
		try: output = config.get(tsec, 'output')
		except Exception as e: _throw('Configuration must specify output', e)

		# Read optional bounds
		try:
			bounds = config.getlist(tsec, 'limits', mapper=float, default=None)
			if bounds and len(bounds) != 2:
				raise ValueError('limits must consist of two floats')
		except Exception as e: _throw('Invalid optional limits', e)

		# Compute random updates to the image
		ns = makeimage(s, mask, bx, nmeas, nrounds, bounds,
				mfilter, bfgs_opts, partial_output, MPI.COMM_WORLD)

		np.save(output, ns.astype(np.float32))

		# Quit the traceloops
		MPI.COMM_WORLD.bcast('ENDTRACE')
	else:
		try:
			# Read element locations
			efiles = matchfiles(config.getlist(tsec, 'elements'))
			elements = ldmats(efiles, nkeys=1)
		except Exception as e: _throw('Configuration must specify elements', e)

		# Load integration parameters
		try: dl = config.get(tsec, 'dl', mapper=float, default=1e-3)
		except Exception as e: _throw('Invalid optional dl', e)
		try: hmax = config.get(tsec, 'hmax', mapper=float, default=0.25)
		except Exception as e: _throw('Invalid optional hmax', e)

		# Load maximum number of path-tracing segments
		try: segmax = config.get(tsec, 'segmax', mapper=int, default=256)
		except Exception as e: _throw('Invalid optional segmax', e)

		# Load path-tracing tolerance
		try: pathtol = config.get(tsec, 'pathtol', mapper=float, default=1e-3)
		except Exception as e: _throw('Invalid optional pathtol', e)


		# Load default background slowness
		try: slowdef = config.get(tsec, 'slowdef', mapper=float, default=None)
		except Exception as e: _throw('Invalid optional slowdef', e)

		# Determine interpolation mode
		linear = config.get(tsec, 'linear', mapper=bool, default=True)
		interp = LinearInterpolator3D if linear else HermiteInterpolator3D

		try:
			# Look for local files and determine local share
			tfiles = matchfiles(config.getlist(tsec, 'timefile'))
			# Determine the local shares of every file
			tfiles = flocshare(tfiles, wcomm)
			# Pull out local share of locally available arrival times
			atimes = dict(kp for tf, (st, ln) in tfiles.iteritems()
					for kp in getatimes(tf, 0, st, ln).iteritems())
		except Exception as e: _throw('Configuration must specify valid timefile', e)

		# The worker communicator is no longer needed
		wcomm.Free()

		# Initiate the worker loop
		traceloop(bx, elements, atimes, dl, hmax, segmax, pathtol,
				slowdef, bfgs_opts, interp, MPI.COMM_WORLD)

	# Keep alive until everybody quits
	MPI.COMM_WORLD.Barrier()
	if not rank: print 'Finished'
