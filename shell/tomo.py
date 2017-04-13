#!/usr/bin/env python

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

'''
Perform an arrival-time-tomographic udpate of sound speeds.
'''

import sys, os

import numpy as np
from numpy.linalg import norm

from scipy.ndimage import median_filter

from math import fsum

from random import sample

from time import time

from itertools import izip

from mpi4py import MPI

from pycwp.cytools.interpolator import HermiteInterpolator3D, LinearInterpolator3D
from pycwp.cytools.regularize import epr, totvar

from habis.pathtracer import PathTracer

from habis.habiconf import HabisConfigParser, HabisConfigError, matchfiles
from habis.formats import loadkeymat as ldkmat, loadmatlist as ldmats
from habis.mpdfile import flocshare
from habis.slowness import Slowness, MaskedSlowness, PiecewiseSlowness


def getatimes(atfile, column=0, filt=None, start=0, stride=1):
	'''
	Read the 2-key arrival-time map with name atfile and remove
	backscatter entries for which key[0] == key[1]. The map is always
	treated as a multi-value map (i.e., 'scalar' is False in loadkeymat).
	The column argument specifies which value index should be retained.

	If filt is not None, it should be a callable which, when called as
	filt(v, t, r) for an arrival time v corresponding to transmit index t
	and receive index r, returns True for a valid measurement and False for
	an invalid measurement. If filt is not specified, all non-backscatter
	measurements are assumed valid.

	Only every stride-th *valid* record, starting with the start-th such
	record, is retained.
	'''
	if not 0 <= start < stride:
		raise ValueError('Index start must be at least zero and less than stride')

	# Load the map, eliminate invalid elemenets, and keep the right portion
	atimes = { }
	idx = 0

	for (t, r), v in ldkmat(atfile, nkeys=2, scalar=False).iteritems():
		# Ignore backscatter and invalid waveforms
		if t == r or (filt and not filt(v, t, r)): continue
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
	def __init__(self, elements, atimes, tracer, 
			slowdef=None, linear=True, comm=MPI.COMM_WORLD):
		'''
		Create a worker, collaborating with other ranks in the given
		communicator comm, to trace optimum paths through slowness maps
		according to the PathTracer instance tracer. Elements have
		coordinates given by element[i] for some index i, and first
		arrival times from transmitter t to receiver r given by
		atimes[t,r].

		Any arrival times corresponding to a (t,r) key for which
		elements[t] or elements[r] is not defined will be discarded.

		Slowness is interpolated with LinearInterpolator3D if linear is
		True (HermiteInterpolator3D otherwise). The interpolator will
		inherit a default (out-of-bounds) slowness slowdef. 
		'''
		# Grab a reference to the path tracer
		self.tracer = tracer

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


	def evaluate(self, s, nm, grad=None):
		'''
		Evaluate a stochastic sample of the cost functional and its
		gradient

		  C(s) = 0.5 * || Ls - D ||**2,
		  grad(C)(s) = L^T [ Ls - D ],

		where L is a stochastic representation of the path-length
		operator mapping the slowness at each cell in s to its
		contribution to the travel times based on the traced paths as
		computed by self.tracer.trace. The data D consists of
		corresponding entries self.atimes[t,r].

		The return value will be (C(s), nr, grad(C)(s)), where nr is
		the number of measurements participating in the sample (this
		may be different than nm if nm < 1, nm > len(self.atimes), or
		if certain paths could not be traced).

		If grad is not None, it should be a floating-point Numpy array
		of shape self.tracer.box.ncell. In this case, gf = grad(C)(s)
		will be stored in the provided array, and the provided array
		will be returned as gf. If grad is None, a new array will be
		allocated for gf. If grad is provided but is not array-like or
		has the wrong shape, a ValueError will be raised.

		The sample consists of nm transmit-receive pairs selected from
		self.atimes with equal probability. If nm >= len(self.atimes)
		or nm < 1, the entire cost functional and gradient will be
		computed. The local sample of the functions is accumulated with
		shares from other MPI ranks in the communicator self.comm using
		reduction operators.

		For path tracing in self.tracer.trace, the slowness s will be
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

		gshape = self.tracer.box.ncell
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
			src, rcv = self.elements[t], self.elements[r]
			try:
				plens, pint = self.tracer.trace(si, src, rcv)
				if not plens or pint <= 0: raise ValueError
			except ValueError:
				# Skip invalid or empty paths
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


def makeimage(cshare, s, nmeas, epochs, updates, beta=0.5, tol=1e-6,
		tvreg=None, mfilter=None, limits=None, partial_output=None):
	'''
	Iteratively compute a slowness image by minimizing the cost functional
	represented in the TomographyTracer instance cshare and using the
	solution to update the given slowness model s (an instance of
	habis.slowness.Slowness or its descendants) defined over the grid
	defined in cshare.tracer.box.

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

	If tvreg is True, the cost function will be regularized with a method
	from pycwp.cytools.regularize. In this case, tvreg should either be a
	scalar regularization parameter used to weight the norm or a kwargs
	dictionary which must contain a 'weight' keyword providing the weight.
	Three optional keywords, 'scale', 'min' and 'every', will be used to
	scale the weight by the float factor 'scale' after every 'every' epochs
	(default: 1) until the weight is no larger than 'min' (default: 0). The
	values of 'every' and 'min' are ignored if 'scale' is not provided. An
	optional 'method' keyword can take the value 'epr' or 'totvar' to
	select the corresponding regularization method from the regularize
	module. If 'method' is not provided, 'totvar' is assumed. Any
	additional keyword arguments are passed to the regularizer.

	After each round of randomized reconstruction, if mfilter is True, a
	median filter of size mfilter will be applied to the image before
	beginning the next round. The argument mfilter can be a scalar or a
	three-element sequence of positive integers.

	If limits is not None, it should be a tuple of the form (slo, shi),
	where slo and shi are, respectively, the lowest and highest allowed
	slowness values. Each update will be clipped to these limits as
	necessary.

	If partial_output is not None, it should be a string specifying a name
	template that will be rendered to store images produced after each
	update. An "update" counts as a update in a single epoch. The formatted
	output name will be partial_output.format(epoch=epoch, iter=iter),
	where "epoch" and "iter" are the epoch index and update iteration
	number, respectively. If partial_output is None, no partial images will
	be stored.
	'''
	# Determine the grid shape
	gshape = cshare.tracer.box.ncell

	if s.shape != gshape: raise ValueError('Shape of s must be %s' % (gshape,))

	# Work arrays
	sp = s.perturb(0)
	gf = np.zeros_like(sp)

	work = np.zeros((s.nnz, 4), dtype=np.float64)
	x = work[:,0]
	lx = work[:,1]
	cg = work[:,2]
	lg = work[:,3]

	# Interpret TV regularization
	tvscale, tvargs = { }, { }
	if tvreg:
		# Make sure a default regularizing method is selected
		tvscale.setdefault('method', totvar)

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

			try:
				# Use a specified regularization method
				method = str(tvargs.pop('method')).strip().lower()
			except KeyError:
				pass
			else:
				if method == 'totvar':
					tvscale['method'] = totvar
				elif method == 'epr':
					tvscale['method'] = epr
				else:
					err = 'Unknown regularization method ' + method
					raise ValueError(err)

			# Optional 'every' and 'min' arguments
			tvscale['every'] = int(tvargs.pop('every', 1))
			tvscale['min'] = float(tvargs.pop('min', 0))

		# Allocate space for expanded update when regularizing
		xp = np.zeros(s.shape, dtype=np.float64)

	def ffg(x):
		'''
		This function returns the (optionally TV regularized) cost
		functional and its gradient for optimization by SGD-BB to
		obtain a contrast update.

		See the TomographyTracer documentation for the general
		(unregularized) form of the cost functional.
		'''
		# Track the run time
		stime = time()

		# Compute the perturbed slowness into sp
		s.perturb(x, sp)

		# Compute the stochastic cost and gradient
		f, nrows, _ = cshare.evaluate(sp, nmeas, grad=gf)

		# Scale cost (and gradient) to mean-squared error
		f /= nrows
		lgf = s.flatten(gf) / nrows

		try:
			tvwt = tvscale['weight']
		except KeyError:
			pass
		else:
			# Unpack update for regularization
			s.unflatten(x, xp)
			tvn, tvng = tvscale['method'](xp, **tvargs)
			f += tvwt * tvn
			lgf += tvwt * s.flatten(tvng)

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
		if limits:
			# Clip the update to the desired range
			s.clip(x, limits[0], limits[1], x)

		if k < 1:
			f, lgf = ffg(x)
			eta = 2 * f / norm(lgf)**2
		elif k >= 2:
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
		if cshare.isRoot: cb = callbackgen(partial_output, s, k, mfilter)
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
	sp = s.perturb(x, sp)
	# Apply a desired filter
	if mfilter: sp[:,:,:] = median_filter(sp, size=mfilter)

	return sp


def callbackgen(templ, s, epoch, mfilter):
	'''
	Build a callback with signature callback(x, nit) to write partial
	images of perturbations x to an assumed slowness model s (as a
	habis.slowness.Slowness instance) for a given SGD-BB epoch 'epoch'.

	If mfilter is True, it should be a value passed as the "size" argument
	to scipy.ndimage.median_filter to smooth the perturbed slowness prior
	to output.

	The callback will store images in npy format with the name given by
	templ.format(epoch=epoch, iter=nit).
	'''
	if not templ: return None

	def callback(x, nit):
		# Write out the iterate
		sp = s.perturb(x)
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

	rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

	# Try to build a tracer from the configuration
	tracer = PathTracer.fromconf(config)
	bx = tracer.box

	if not rank:
		print 'Solution defined on grid', bx.lo, bx.hi, bx.ncell

	tsec = 'tomo'

	def _throw(msg, e, sec=None):
		if not sec: sec = tsec
		raise HabisConfigError.fromException(msg + ' in [%s]' % (sec,), e)


	# Load the slowness (may be a value or a file name)
	try: s = config.get(tsec, 'slowness')
	except Exception as e: _throw('Configuration must specify slowness', e)

	# Determine whether piecewise-constant slowness models are desired
	try: piecewise = config.get(tsec, 'piecewise', mapper=bool, default=False)
	except Exception as e: _throw('Invalid optional piecewise')

	try:
		# Load a slowness mask, if one is specified
		mask = config.get(tsec, 'slowmask', default=None)
		if mask: mask = np.load(mask).astype(int if piecewise else bool)
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

	try:
		# Read element locations
		efiles = matchfiles(config.getlist(tsec, 'elements'))
		elements = ldmats(efiles, nkeys=1)
	except Exception as e: _throw('Configuration must specify elements', e)

	# Load default background slowness
	try: slowdef = config.get(tsec, 'slowdef', mapper=float, default=None)
	except Exception as e: _throw('Invalid optional slowdef', e)

	# Determine interpolation mode
	linear = config.get(tsec, 'linear', mapper=bool, default=True)

	try: limits = config.getlist(tsec, 'limits', mapper=float, default=None)
	except Exception as e: _throw('Configuration must specify valid limits')

	if limits:
		if len(limits) != 2:
			_throw('Optional limits must be a two-element list')
		# Define a validity filter for arrival times
		limits = sorted(limits)
		if not rank: print 'Restricting arrival times to average slowness in', limits
		def atfilt(v, t, r):
			# Sanity check
			if t == r: return False
			# Find average straight-ray slowness, if possible
			try: aslw = v / norm(elements[t] - elements[r])
			except KeyError: return False
			return limits[0] <= aslw <= limits[1]
	else: atfilt = None

	try:
		# Look for local files and determine local share
		tfiles = matchfiles(config.getlist(tsec, 'timefile'))
		# Determine the local shares of every file
		tfiles = flocshare(tfiles, MPI.COMM_WORLD)
		# Pull out local share of locally available arrival times
		atimes = dict(kp for tf, (st, ln) in tfiles.iteritems()
				for kp in getatimes(tf, 0, atfilt, st, ln).iteritems())
	except Exception as e: _throw('Configuration must specify valid timefile', e)

	# Build the cost calculator
	cshare = TomographyTracer(elements, atimes, tracer,
					slowdef, linear, MPI.COMM_WORLD)

	if piecewise:
		if mask is None: raise ValueError('Slowness mask is required in piecewise mode')
		slw = PiecewiseSlowness(mask, s)
		if not rank: print 'Assuming piecewise model with values', slw._s
	else:
		# Convert the scalar slowness or file name into a matrix
		try: s = float(s)
		except (ValueError, TypeError): s = np.load(s).astype(np.float64)
		else: s = s * np.ones(bx.ncell, dtype=np.float64)

		if mask is not None: slw = MaskedSlowness(s, mask)
		else: slw = Slowness(s)

	# Compute random updates to the image
	ns = makeimage(cshare, slw, nmeas, epochs, updates, beta,
			tol, tvreg, mfilter, limits, partial_output)

	if not rank:
		np.save(output, ns.astype(np.float32))
		print 'Finished'
