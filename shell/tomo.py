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

from pycwp.cytools.boxer import Box3D
from pycwp.cytools.interpolator import HermiteInterpolator3D, LinearInterpolator3D
from pycwp.cytools.regularize import totvar

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


class TomographyTracer(object):
	'''
	A class encapsulating a single MPI rank that participates in computing
	path tracing for arrival-time tomography. Each instance takes
	responsibility for the set of arrival-time measurements provided to it.
	'''
	def __init__(self, box, elements, atimes, ptol=1e-3,
			itol=1e-5, segmax=256, optargs={ },
			slowdef=None, linear=True, comm=MPI.COMM_WORLD):
		'''
		Create a worker collaborating with other ranks in the given
		communicator comm. The image is defined over the Box3D instance
		box, with element locations given by element[i] for some index
		i, and first arrival times from transmitter t to receiver r
		given by atimes[t,r].

		Any arrival times corresponding to a (t,r) key for which
		elements[t] or elements[r] is not defined will be discarded.

		When tracing propagation paths, individual path integrals are
		computing adaptively with a tolerance of itol, while path
		optimization proceeds with a tolerance of ptol and a maximum
		segment count segmax.

		Slowness is interpolated with LinearInterpolator3D if linear is
		True (HermiteInterpolator3D otherwise). The interpolator will
		inherit a default (out-of-bounds) slowness slowdef. The optargs
		kwargs dictionary will be passed to Interpolator3D.minpath to
		control the optimization process.
		'''
		# Make a copy of the image box
		self.box = Box3D(box.lo, box.hi)
		self.box.ncell = box.ncell

		# Make a copy of the element and arrival-time maps
		self.elements = { }
		self.atimes = { }

		# Record the MPI communicator
		self.comm = comm

		for (t, r), v in atimes.iteritems():
			# The transmit and receive indices must be distinct
			if t == r: continue

			try:
				# Ensure this arrival time has element locations
				et = elements[t]
				er = elements[r]
			except KeyError:
				pass
			else:
				self.elements[t] = et
				self.elements[r] = er
				self.atimes[t,r] = v

		# Record solver parameters
		self.ptol = float(ptol)
		self.itol = float(itol)
		self.nmax = int(segmax)
		self.optargs = dict(optargs)
		self.slowdef = slowdef
		self.interpolator = (LinearInterpolator3D if linear
					else HermiteInterpolator3D)

	@property
	def isRoot(self):
		'''
		True if this instance has a communicator and is rank 0 of its
		communicator, False otherwise.
		'''
		return self.comm and not self.comm.rank


	def pathtrace(self, si, t, r):
		'''
		Given an interpolated slowness map si (as Interpolator3D), a
		transmitter t and a receiver r in world coordinates, trace an
		optimum path from t to r using

		  si.minpath(gt, gr, self.nmax, self.itol,
				  self.ptol, self.box.cell, **self.optargs),

		where gt, gr are the grid coordinates of t and r, respectively,
		according to self.box.cart2cell.

		The determined path will be marched through self.box to produce a
		map from cell indices in the box to the length of the intersection of
		that cell and the path (i.e., the map is a path-length matrix for a
		single path).

		The resulting path-length map will be returned, along with the
		path-length integral si.pathint(path, self.itol, self.cell) for
		the optimum path.

		In case any failure prevents the determination of a convergent,
		optimized path-length map, an empty map and a zero path-length
		integral will be returned.
		'''
		box = self.box

		# Compute the minimum path (if possible) or return an empty path
		gsrc = box.cart2cell(*self.elements[t])
		grcv = box.cart2cell(*self.elements[r])

		try:
			popt, pint = si.minpath(gsrc, grcv, self.nmax, self.itol,
						self.ptol, box.cell, **self.optargs)
		except ValueError:
			return { }, 0.0

		# Convert path to Cartesian coordinates and march segments
		points = np.array([box.cell2cart(*p) for p in popt])
		try: marches = box.raymarcher(points)
		except ValueError as e:
			print 'NOTE: ray march failed:', (t, r), str(e)
			return { }, 0.0

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
		# Return the path map and the path-length integral
		return { k: fsum(v) for k, v in plens.iteritems() }, pint


	def evaluate(self, s, nm, grad=None):
		'''
		Evaluate a stochastic sample of the cost functional and its
		gradient

		  C(s) = 0.5 * || Ls - D ||**2,
		  grad(C)(s) = L^T [ Ls - D ],

		where L is a stochastic representation of the path-length
		operator mapping the slowness at each cell in s to its
		contribution to the travel times based on the traced paths as
		computed by self.pathtrace. The data D consists of
		corresponding entries self.atimes[t,r].

		The return value will be (C(s), nr, grad(C)(s)), where nr is
		the number of measurements participating in the sample (this
		may be different than nm if nm < 1, nm > len(self.atimes), or
		if certain paths could not be traced).

		If grad is not None, it should be a floating-point Numpy array
		of shape self.box.ncell. In this case, gf = grad(C)(s) will be
		stored in the provided array, and the provided array will be
		returned as gf. If grad is None, a new array will be allocated
		for gf. If grad is provided but is not array-like or has the
		wrong shape, a ValueError will be raised.

		The sample consists of nm transmit-receive pairs selected from
		self.atimes with equal probability. If nm >= len(self.atimes)
		or nm < 1, the entire cost functional and gradient will be
		computed. The local sample of the functions is accumulated with
		shares from other MPI ranks in the communicator self.comm using
		reduction operators.

		For path tracing in self.pathtrace, the slowness s will be
		interpolated as si = self.interpolator(s).
		'''
		rank, size = self.comm.rank, self.comm.size

		# Interpolate the slowness
		si = self.interpolator(s)
		si.default = self.slowdef

		# Compute the random sample of measurements
		if nm < 1 or nm > len(self.atimes): nm = len(self.atimes)
		trset = sample(atimes.keys(), nm)

		# Accumulate the local cost function and gradient
		f = []

		gshape = self.box.ncell
		if grad is not None:
			# Use a provided gradient, if possible
			try:
				gsh = grad.shape
				gdtype = grad.dtype
			except AttributeError:
				raise ValueError('Argument "grad" must be None or Numpy-array compatible')

			if gshape != gsh or not np.issubdtype(gdtype, np.inexact):
				raise ValueError('Non-None "grad" must be a floating-point array of shape %s' % (gshape,))

			grad[:,:,:] = 0.
			gf = grad
		else: gf = np.zeros(gshape, dtype=np.float64, order='C')

		nrows, nskip = 0L, 0L

		# Compute contributions for each source-receiver pair
		for t, r in trset:
			plens, pint = self.pathtrace(si, t, r)
			if not plens or pint <= 0:
				# Skip bad paths
				nskip += 1
				continue
			nrows += 1
			# Calculate error in model arrival time
			err = pint - atimes[t,r]
			f.append(err**2)
			# Add gradient contribution
			for c, l in plens.iteritems(): gf[c] += l * err

		if nskip: print 'MPI rank %d of %d: skipped %d untraceable paths' % (rank, size, nskip)

		# Accumulate the cost functional and row count
		f = self.comm.allreduce(0.5 * fsum(f), op=MPI.SUM)
		nrows = self.comm.allreduce(nrows, op=MPI.SUM)
		# Use the lower-level routine for the in-place gradient accumulation
		self.comm.Allreduce(MPI.IN_PLACE, gf, op=MPI.SUM)

		return f, nrows, gf


def makeimage(cshare, s, mask, nmeas, epochs, updates, beta=0.5,
		tol=1e-6, tvreg=None, mfilter=None, partial_output=None):
	'''
	Iteratively compute a slowness image by minimizing the cost functional
	represented in the TomographyTracer instance cshare and using the
	solution to update the given profile s defined over the grid defined in
	cshare.box.

	If mask is not None, it should be a bool array with the same shape as s
	(and cshare.box.ncell) that is True for cells that should be updated
	and False for cells that should remain unchanged.

	The Stochastic Gradient Descent, Barzilai-Borwein (SGB-BB) method of
	Tan, et al. (2016) is used to compute the image. The method continues
	for at most 'epochs' epochs, with a total of 'updates' stochastic
	descent steps per epoch. A single stochastic descent is made by
	sampling the global cost functional (mean-squared arrival-time
	error) using nmeas measurements per MPI rank.

	The descent step is selecting using a stochastic Barzilai-Borwein (BB)
	scheme. The first two epochs will each use a fixed step size of 1.
	Later updates rely on approximations to the gradient in previous
	epochs. The approximate gradient at epoch k is defined recursively over
	t updates as

		g_{k,t} = beta * grad(f_t)(x_t) + (1 - beta) g_{k,t-1},

	where g_{k,0} == 0, f_t is the t-th sampled cost functional for the
	epoch and x_t is the solution at update t.

	If tvreg is True, the cost function will be regularized with the
	total-variation norm from pycwp.cytools.regularize.totvar. In this
	case, tvreg should either be a scalar regularization parameter used to
	weight the norm or a kwargs dictionary which must contain a 'weight'
	keyword providing the weight. Three optional keywords, 'scale', 'min'
	and 'every', will be used to scale the weight by the float factor
	'scale' after every 'every' epochs (default: 1) until the weight is no
	larger than 'min' (default: 0). The values of 'every' and 'min' are
	ignored if 'scale' is not provided. Any additional keyword arguments
	are passed to totvar.

	After each round of randomized reconstruction, if mfilter is True, a
	median filter of size mfilter will be applied to the image before
	beginning the next round. The argument mfilter can be a scalar or a
	three-element sequence of positive integers.

	If partial_output is not None, it should be a string specifying a name
	template that will be rendered to store images produced after each
	update. An "update" counts as a update in a single epoch. The formatted
	output name will be partial_output.format(epoch=epoch, iter=iter),
	where "epoch" and "iter" are the epoch index and update iteration
	number, respectively. If partial_output is None, no partial images will
	be stored.

	Participating ranks will be pulled from the given MPI communicator,
	wherein rank 0 must invoke this function and all other ranks must
	invoke traceloop().
	'''
	# Determine the grid shape
	gshape = cshare.box.ncell

	# Make sure the image is the right shape
	s = np.array(s, dtype=np.float64)

	if mask is None: mask = np.ones(s.shape, dtype=bool)
	else: mask = np.asarray(mask, dtype=bool)

	if s.shape != gshape or s.shape != mask.shape:
		raise ValueError('Shape of s and optional mask must be %s' % (gshape,))

	nnz = np.sum(mask.astype(np.int64))

	# Work arrays
	sp = np.copy(s)
	gf = np.zeros_like(s)

	work = np.zeros((nnz,4), dtype=np.float64)
	x = work[:,0]
	lx = work[:,1]
	cg = work[:,2]
	lg = work[:,3]

	# Interpret TV regularization
	tvscale, tvargs = { }, { }
	if tvreg:
		try:
			# Treat the tvreg argument as a simple float
			tvscale['weight'] = float(tvreg)
		except TypeError:
			# Non-numeric tvreg should be a dictionary
			tvargs = dict(tvreg)

			# Required weight parameter
			tvscale['weight'] = float(tvargs.pop('weight'))

			# Optional 'scale' argument
			try: tvscale['scale'] = float(tvargs.pop('scale'))
			except KeyError: pass

			# Optional 'every' and 'min' arguments
			tvscale['every'] = int(tvargs.pop('every', 1))
			tvscale['min'] = float(tvargs.pop('min', 0))

	if 'weight' in tvscale:
		xp = np.zeros_like(s)

	def ffg(x):
		'''
		This function returns the (optionally TV regularized) cost
		functional and its gradient for optimization by SGD-BB to
		obtain a contrast update.

		See the traceloop documentation for the general (unregularized)
		form of the cost functional.
		'''
		# Track the run time
		stime = time()

		# Compute the perturbed slowness
		sp[mask] = s[mask] + x

		# Compute the stochastic cost and gradient
		f, nrows, _ = cshare.evaluate(sp, nmeas, grad=gf)

		# Scale cost (and gradient) to mean-squared error
		f /= nrows
		lgf = gf[mask] / nrows

		try:
			tvwt = tvscale['weight']
		except KeyError:
			pass
		else:
			# Unpack update for regularization
			xp[mask] = x
			tvn, tvng = totvar(xp, **tvargs)
			f += tvwt * tvn
			lgf += tvwt * tvng[mask]

		# The time to evaluate the function and gradient
		stime = time() - stime

		if cshare.isRoot: print 'Cost evaluation time: %0.2f sec' % (stime,)

		return f, lgf

	# Step smoothing coefficient
	lck = 0.0

	# For convergence testing
	maxcost = 0.0
	converged = False

	for k in range(epochs):
		if k < 2:
			eta = 1.0
		else:
			# Compute change in solution and gradient
			lx += x
			lg += cg

			nlx = norm(lx)**2
			xdg = abs(np.dot(lx, lg))

			if xdg < sys.float_info.epsilon * nlx:
				# Terminate if step size blows up
				if cshare.isRoot:
					print 'TERMINATE: epoch', k, 'step size breakdown'
				break

			eta = nlx / xdg / updates

			# Smooth the step (use logs to avoid overflow)
			lkp = np.log(k + 1.0)
			lck = ((k - 2) * lck + np.log(eta) + lkp) / (k - 1.0)
			eta = np.exp(lck - lkp)

		if cshare.isRoot:
			print 'Epoch', k,
			if tvscale: print 'TV weight', tvscale['weight'],
			print 'gradient descent step', eta

		# Copy negative of last solution and gradient
		lx[:] = -x
		lg[:] = -cg

		# Clear gradient for next iteration
		cg[:] = 0

		# Build a callback to write per-iteration results, if desired
		if cshare.isRoot: cb = callbackgen(partial_output, s, mask, k, mfilter)
		else: cb = None

		for t in range(updates):
			# Compute the sampled cost functional and its gradient
			f, lgf = ffg(x)

			# Print some convergence numbers
			if cshare.isRoot: print 'At epoch', k, 'update', t, 'cost', f

			# Adjust the solution against the gradient
			x -= eta * lgf

			# Store the partial update if desired
			if cb: cb(x, t)

			# Check for convergence
			maxcost = max(f, maxcost)
			if f < tol * maxcost:
				converged = True
				break

			# Update the average gradient
			cg[:] = beta * lgf + (1 - beta) * cg

		if converged:
			if cshare.isRoot: print 'TERMINATE: Convergence achieved'
			break

		# Adjust the regularization weight as appropriate
		if ('scale' in tvscale and not (k + 1) % tvscale['every']
				and tvscale['weight'] > tvscale['min']):
			tvscale['weight'] *= tvscale['scale']

	# Update the image
	s[mask] += x

	# Apply a desired filter
	if mfilter: s[:,:,:] = median_filter(s, size=mfilter)

	return s


def callbackgen(templ, s, mask, epoch, mfilter):
	'''
	Build a callback with signature callback(x, nit) to write partial
	images of perturbations x to an assumed slowness s, with the given
	update mask (True where samples will be perturbed, False elsewhere),
	for a given SGD-BB epoch 'epoch'.

	If mfilter is True, it should be a value passed as the "size" argument
	to scipy.ndimage.median_filter to smooth the perturbed slowness prior
	to output.

	The callback will store images in npy format with the name given by
	templ.format(epoch=epoch, iter=nit).
	'''
	if not templ: return None

	def callback(x, nit):
		# Write out the iterate
		sp = s.copy()
		sp[mask] += x
		if mfilter: sp[:,:,:] = median_filter(sp, size=mfilter)
		fname = templ.format(epoch=epoch, iter=nit)
		np.save(fname, sp.astype(np.float32))

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

	tsec = 'tomo/grid'

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

	if not rank:
		print 'Solution defined on grid', bx.lo, bx.hi, bx.ncell

	tsec = 'tomo/sgd'

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
		# Read the number of epochs
		epochs = config.get(tsec, 'epochs', mapper=int, default=1)
		if epochs < 1: raise ValueError('epochs must be positive')
	except Exception as e: _throw('Invalid optional epochs', e)

	try:
		# Read the number of updates per epoch
		updates = config.get(tsec, 'updates', mapper=int, default=1)
		if updates < 1: raise ValueError('updates must be positive')
	except Exception as e: _throw('Invalid optional updates', e)

	try:
		# Read the number sampled gradient approximation weight
		beta = config.get(tsec, 'beta', mapper=float, default=0.5)
		if not 0 < beta <= 1.0: raise ValueError('beta must be in range (0, 1]')
	except Exception as e: _throw('Invalid optional beta', e)

	try:
		# Read the number sampled gradient approximation weight
		tol = config.get(tsec, 'tol', mapper=float, default=1e-6)
		if not tol > 0: raise ValueError('tol must be positive')
	except Exception as e: _throw('Invalid optional tol', e)

	try:
		# Read optional total-variation regularization parameter
		tvreg = config.get(tsec, 'tvreg', mapper=dict,
					checkmap=False, default=None)
	except Exception as e: _throw('Invalid optional tvreg', e)

	try:
		# Read optional the filter parameters
		mfilter = config.get(tsec, 'mfilter', mapper=int, default=None)
	except Exception as e: _throw('Invalid optional mfilter', e)

	# Read optional partial_output format string
	partial_output = config.get(tsec, 'partial_output', default=None)

	# Read required final output
	try: output = config.get(tsec, 'output')
	except Exception as e: _throw('Configuration must specify output', e)

	tsec = 'tomo/tracer'

	# Read optional bfgs option dictionary for path tracing
	try:
		bfgs_opts = config.get(tsec, 'optimizer',
				mapper=dict, checkmap=False, default={ })
	except Exception as e: _throw('Invalid optional optimizer', e)

	try:
		# Read element locations
		efiles = matchfiles(config.getlist(tsec, 'elements'))
		elements = ldmats(efiles, nkeys=1)
	except Exception as e: _throw('Configuration must specify elements', e)

	# Load integration tolerance
	try: pintol = config.get(tsec, 'pintol', mapper=float, default=1e-5)
	except Exception as e: _throw('Invalid optional pintol', e)

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

	try:
		# Look for local files and determine local share
		tfiles = matchfiles(config.getlist(tsec, 'timefile'))
		# Determine the local shares of every file
		tfiles = flocshare(tfiles, MPI.COMM_WORLD)
		# Pull out local share of locally available arrival times
		atimes = dict(kp for tf, (st, ln) in tfiles.iteritems()
				for kp in getatimes(tf, 0, st, ln).iteritems())
	except Exception as e: _throw('Configuration must specify valid timefile', e)

	# Build the cost calculator
	cshare = TomographyTracer(bx, elements, atimes, pathtol,
			pintol, segmax, bfgs_opts, slowdef, linear, MPI.COMM_WORLD)

	# Compute random updates to the image
	ns = makeimage(cshare, s, mask, nmeas, epochs, updates,
			beta, tol, tvreg, mfilter, partial_output)

	if not rank:
		np.save(output, ns.astype(np.float32))
		print 'Finished'
