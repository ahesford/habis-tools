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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, norm as snorm

from math import fsum

from random import sample

from time import time

from itertools import izip

from mpi4py import MPI

from pycwp.cytools.interpolator import HermiteInterpolator3D, LinearInterpolator3D
from pycwp.cytools.interpolator import TraceError
from pycwp.cytools.regularize import epr, totvar, tikhonov
from pycwp.iterative import lsmr
from pycwp.cytools.boxer import Segment3D

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


class TomographyTask(object):
	'''
	A root class to encapsulate a single MPI rank that participates in
	tomographic reconstructions. Each instances takes responsibility for a
	set of arrival-time measurements provided to it.
	'''
	def __init__(self, elements, atimes, comm=MPI.COMM_WORLD):
		'''
		Create a worker that collaborates with other ranks in the given
		MPI communicator comm to compute tomographic images. Elements
		have coordinates given by elements[i] for some integer index i,
		and first-arrival times from transmitter t ot receiver r are
		given by atimes[t,r].

		Any arrival times corresponding to a (t, r) pair for which
		elements[t] or elements[r] is not defined, or for which t == r,
		will be discarded.
		'''
		# Record the communicator
		self.comm = comm

		# Make a copy of the element and arrival-time maps
		self.elements = { }
		self.atimes = { }

		for (t, r), v in atimes.iteritems():
			# Make sure t-r indices are distinct
			if t == r: continue

			try:
				# Ensure arrival time has element locations
				et = elements[t]
				er = elements[r]
			except KeyError:
				pass
			else:
				self.elements[t] = et
				self.elements[r] = er
				self.atimes[t,r] = v

	@property
	def isRoot(self):
		'''
		True if this instance has a communicator and is rank 0 in that
		communicator, False otherwise.
		'''
		return self.comm and not self.comm.rank


class StraightRayTracer(TomographyTask):
	'''
	A subclass of TomographyTask to perform straight-ray path tracing.
	'''
	def __init__(self, elements, atimes, box, fresnel=None, comm=MPI.COMM_WORLD):
		'''
		Create a worker for straight-ray tomography on the element
		pairs present in atimes. The imaging grid is represented by the
		Box3D box.

		If Fresnel is not None, it should be a positive floating-point
		value. In this case, rays are represented as ellipsoids that
		embody the first Fresnel zone for a principal wavelength of
		fresnel (in units that match the units of box.cell).
		'''
		# Initialize the data and communicator
		super(StraightRayTracer, self).__init__(elements, atimes, comm)

		# Grab a reference to the box
		self.box = box
		ncell = self.box.ncell

		# Build the path-length matrix and RHS vector
		indptr = [ 0 ]
		indices = [ ]
		data = [ ]
		rhs = [ ]
		for (t, r), v in sorted(self.atimes.iteritems()):
			# March straight through the grid
			seg = Segment3D(self.elements[t], self.elements[r])
			# Compute the total segment length
			if not fresnel:
				# Build a simple row of the path-length matrix
				path = self.box.raymarcher(seg)
				for cidx, (s, e) in path.iteritems():
					data.append(seg.length * (e - s))
					indices.append(np.ravel_multi_index(cidx, ncell))
			else:
				# Build a row based on the first Fresnel zone
				hits = self.box.fresnel(seg, fresnel)
				htot = fsum(hits.itervalues())
				for cidx, ll in hits.iteritems():
					data.append(seg.length * ll / htot)
					indices.append(np.ravel_multi_index(cidx, ncell))
			indptr.append(len(indices))
			rhs.append(v)

		self.pathmat = csr_matrix((data, indices, indptr),
				shape=(len(rhs), np.prod(ncell)), dtype=np.float64)
		self.rhs = np.array(rhs, dtype=np.float64)

	def lsmr(self, s, **kwargs):
		'''
		Iteratively compute, using LSMR, a slowness image that
		satisfies the straight-ray arrival-time equations implicit in
		this StraightRayTracer instance. The solution is represented as
		a perturbation to the slowness s, an instance of
		habis.slowness.Slowness or its descendants, defined on the grid
		self.box.

		The LSMR implementation uses pycwp.iterative.lsmr to support
		arrival times distributed across multiple MPI tasks in
		self.comm. The keyword arguments kwargs will be passed to
		lsmr to customize the solution process. Forbidden keyword
		arguments are "A", "b", "unorm" and "vnorm".

		If a special keyword argument, 'coleq', is True, the columns of
		the path-length operator will be scaled to that they all have
		unity norm. The value of 'coleq' is False by default. The
		'coleq' keyword argument will not be provided to
		pycwp.iterative.lsmr.

		The return value will be the final, perturbed solution.
		'''
		if not self.isRoot:
			# Make sure non-root tasks are always silent
			kwargs['show'] = False

		if 'calc_var' in kwargs:
			raise TypeError('Argument "calc_var" is forbidden')

		coleq = kwargs.pop('coleq', False)

		ncell = self.box.ncell

		# RHS is arrival times minus unperturbed solution
		rhs = self.rhs - self.pathmat.dot(s.perturb(0).ravel('C'))

		# Composite slowness transform and path-length operator as CSR
		pathmat = self.pathmat.dot(s.tosparse()).tocsr()

		if coleq:
			# Compute norms of columns of global matrix
			colscale = snorm(pathmat, axis=0)**2
			self.comm.Allreduce(MPI.IN_PLACE, colscale, op=MPI.SUM)
			np.sqrt(colscale, colscale)

			# Clip normalization factors to avoid blow-up
			mxnrm = np.max(colscale)
			np.clip(colscale, 1e-3 * mxnrm, mxnrm, colscale)

			# Normalize the columns
			pathmat.data /= np.take(colscale, pathmat.indices)
		else:
			colscale = 1.0

		# Transpose operation requires communications
		def amvp(u):
			# Synchronize
			self.comm.Barrier()
			# Multiple by transposed local share, then flatten
			v = pathmat.T.dot(u)
			# Accumulate contributions from all ranks
			self.comm.Allreduce(MPI.IN_PLACE, v, op=MPI.SUM)
			return v
		A = LinearOperator(shape=pathmat.shape,
				matvec=pathmat.dot, rmatvec=amvp, dtype=pathmat.dtype)

		def unorm(u):
			# Synchronize
			self.comm.Barrier()
			# Norm of distributed vectors, to all ranks
			un = norm(u)**2
			return np.sqrt(self.comm.allreduce(un, op=MPI.SUM))

		results = lsmr(A, rhs, unorm=unorm, vnorm=norm, **kwargs)
		if self.isRoot:
			print 'LSMR terminated after %d iterations for reason %d' % (results[2], results[1])

		# Correct column scaling in perturbation solution
		return s.perturb(results[0] / colscale)


class BentRayTracer(TomographyTask):
	'''
	A subclass of TomographyTask to perform bent-ray path tracing.
	'''
	def __init__(self, elements, atimes, tracer, fresnel=None,
			slowdef=None, linear=True, comm=MPI.COMM_WORLD):
		'''
		Create a worker to do path tracing on the element pairs present
		in atimes. Slowness will be interpolated with
		LinearInterpolator3D if linear is True (HermiteInterpolator3D
		otherwise). The interpolator will inherit a default
		(out-of-bounds) slowness slowdef.
		'''
		# Initialize the data and communicator
		super(BentRayTracer, self).__init__(elements, atimes, comm)

		# Grab a reference to the path tracer
		self.tracer = tracer
		# Copy the Fresnel parameter
		self.fresnel = fresnel

		# Record solver parameters
		self.slowdef = slowdef
		self.interpolator = (LinearInterpolator3D if linear
					else HermiteInterpolator3D)

	def evaluate(self, s, nm):
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

		# The Fresnel wavelength parameter (or None)
		fresnel = self.fresnel

		# Compute the random sample of measurements
		if nm < 1 or nm > len(self.atimes): nm = len(self.atimes)
		trset = sample(atimes.keys(), nm)

		# Accumulate the local cost function and gradient
		f = []

		gshape = self.tracer.box.ncell
		gf = np.zeros(gshape, dtype=np.float64, order='C')

		nrows, nskip = 0L, 0L

		# Compute contributions for each source-receiver pair
		for t, r in trset:
			src, rcv = self.elements[t], self.elements[r]
			try:
				plens, pint = self.tracer.trace(si, src, rcv, fresnel)
				if not plens or pint <= 0: raise ValueError
			except (ValueError, TraceError):
				# Skip invalid or empty paths
				nskip += 1
				continue

			nrows += 1
			# Calculate error in model arrival time
			err = pint - atimes[t,r]
			f.append(err**2)
			# Add gradient contribution
			for c, l in plens.iteritems(): gf[c] += l * err

		# Accumulate the cost functional and row count
		f = self.comm.allreduce(0.5 * fsum(f), op=MPI.SUM)
		nrows = self.comm.allreduce(nrows, op=MPI.SUM)
		# Use the lower-level routine for the in-place gradient accumulation
		self.comm.Allreduce(MPI.IN_PLACE, gf, op=MPI.SUM)

		return f, nrows, gf

	@staticmethod
	def callbackgen(templ, s, epoch, mfilter):
		'''
		Build a callback with signature callback(x, nit) to write
		partial images of perturbations x to an assumed slowness model
		s (as a habis.slowness.Slowness instance) for a given SGD-BB
		epoch 'epoch'.

		If mfilter is True, it should be a value passed as the "size"
		argument to scipy.ndimage.median_filter to smooth the perturbed
		slowness prior to output.

		The callback will store images in npy format with the name
		given by templ.format(epoch=epoch, iter=nit).
		'''
		if not templ: return None

		def callback(x, nit):
			# Write out the iterate
			sp = s.perturb(x)
			if mfilter: sp[:,:,:] = median_filter(sp, size=mfilter)
			fname = templ.format(epoch=epoch, iter=nit)
			np.save(fname, sp.astype(np.float32))

		return callback

	def sgd(self, s, nmeas, epochs, updates, beta=0.5, tol=1e-6,
			regularizer=None, mfilter=None,
			limits=None, partial_output=None):
		'''
		Iteratively compute a slowness image by minimizing the cost
		functional represented in this BentRayTracer instance and using
		the solution to update the given slowness model s (an instance
		of habis.slowness.Slowness or its descendants) defined over the
		grid defined in self.tracer.box.

		The Stochastic Gradient Descent, Barzilai-Borwein (SGB-BB)
		method of Tan, et al. (2016) is used to compute the image. The
		method continues for at most 'epochs' epochs, with a total of
		'updates' stochastic descent steps per epoch. A single
		stochastic descent is made by sampling the global cost
		functional (mean-squared arrival-time error) using nmeas
		measurements per MPI rank.

		The descent step is selecting using a stochastic
		Barzilai-Borwein (BB) scheme. The first two epochs will each
		use a fixed step size of 1. Later updates rely on
		approximations to the gradient in previous epochs. The
		approximate gradient at epoch k is defined recursively over t
		updates as

		  g_{k,t} = beta * grad(f_t)(x_t) + (1 - beta) g_{k,t-1},

		where g_{k,0} == 0, f_t is the t-th sampled cost functional for
		the epoch and x_t is the solution at update t.

		If regularizer is not None, the cost function will be
		regularized with a method from pycwp.cytools.regularize. The
		value of regularizer must be None or a kwargs dictionary that
		contains at least a 'weight' keyword that provides a scalar
		weight for the regularization term. Three optional keywords,
		'scale', 'min' and 'every', will be used to scale the weight by
		the float factor 'scale' after every 'every' epochs (default:
		1) until the weight is no larger than 'min' (default: 0). The
		values of 'every' and 'min' are ignored if 'scale' is not
		provided. An optional 'method' keyword can take the value
		'epr', 'totvar' or 'tikhonov' to select the corresponding
		regularization method from the regularize module. If 'method'
		is not provided, 'totvar' is assumed. Any remaining keyword
		arguments are passed through to the regularizer.

		After each round of randomized reconstruction, if mfilter is
		True, a median filter of size mfilter will be applied to the
		image before beginning the next round. The argument mfilter can
		be a scalar or a three-element sequence of positive integers.

		If limits is not None, it should be a tuple of the form (slo,
		shi), where slo and shi are, respectively, the lowest and
		highest allowed slowness values. Each update will be clipped to
		these limits as necessary.

		If partial_output is not None, it should be a string specifying
		a name template that will be rendered to store images produced
		after each update. An "update" counts as a update in a single
		epoch. The formatted output name will be
		partial_output.format(epoch=epoch, iter=iter), where "epoch"
		and "iter" are the epoch index and update iteration number,
		respectively. If partial_output is None, no partial images will
		be stored.
		'''
		# Determine the grid shape
		gshape = self.tracer.box.ncell

		if s.shape != gshape:
			raise ValueError('Shape of s must be %s' % (gshape,))

		# Interpret TV regularization
		rgwt, rgfunc, rgargs = { }, None, { }
		if regularizer:
			# Copy the regularizer dictionary
			rgargs = dict(regularizer)

			# Make sure a weight is specified
			try:
				rgwt['weight'] = float(rgargs.pop('weight'))
			except KeyError:
				raise KeyError('Regularizer must specify a weight')
			if rgwt['weight'] <= 0:
				raise ValueError('Regularizer weight must be positive')

			# Pull the desired regularization function
			rgname = rgargs.pop('method', 'totvar').strip().lower()
			try:
				rgfunc = { 'totvar': totvar, 'epr': epr,
						'tikhonov': tikhonov, }[rgname]
			except KeyError:
				raise NameError('Unrecognized regularization method "%s"' % (rgname,))

			# Optional 'scale' argument
			try: rgwt['scale'] = float(rgargs.pop('scale'))
			except KeyError: pass

			# Optional 'every' and 'min' arguments
			rgwt['every'] = int(rgargs.pop('every', 1))
			rgwt['min'] = float(rgargs.pop('min', 0))

			# Allocate space for expanded update when regularizing
			xp = np.zeros(s.shape, dtype=np.float64)

		# Track whether paths have been skipped in each iteration
		maxrows = nmeas * self.comm.size

		def ffg(x, epoch, update):
			'''
			This function returns the (optionally TV regularized) cost
			functional and its gradient for optimization by SGD-BB to
			obtain a contrast update.

			See the BentRayTracer documentation for the general
			(unregularized) form of the cost functional.
			'''
			# Track the run time
			stime = time()

			# Compute the perturbed slowness into sp
			sp = s.perturb(x)

			# Compute the stochastic cost and gradient
			f, nrows, gf = self.evaluate(sp, nmeas)

			if nrows:
				# Scale cost (and gradient) to mean-squared error
				f /= nrows
				lgf = s.flatten(gf) / nrows

			# Calculate RMS data error
			rmserr = np.sqrt(2. * f)

			if rgfunc:
				rw = rgwt['weight']
				# Unpack update for regularization
				xp = s.unflatten(x)
				rgn, rgng = rgfunc(xp, **rgargs)
				f += rw * rgn
				lgf += rw * s.flatten(rgng)

			# The time to evaluate the function and gradient
			stime = time() - stime

			if self.isRoot and epoch >= 0 and update >= 0:
				# Print some convergence numbers
				msgfmt = ('At epoch %d update %d cost %#0.4g '
						'RMSE %#0.4g (time: %0.2f sec)')
				print msgfmt % (epoch, update, f, rmserr, stime)
				if nrows != maxrows:
					print '%d/%d untraceable paths' % (maxrows - nrows, maxrows)

			return f, lgf

		# Step smoothing coefficient
		lck = 0.0

		# For convergence testing
		maxcost = 0.0
		converged = False

		# The first guess at a solution and its gradient
		x = np.zeros((s.nnz,), dtype=np.float64)
		cg = np.zeros((s.nnz,), dtype=np.float64)

		for k in range(epochs):
			if limits:
				# Clip the update to the desired range
				x = s.clip(x, limits[0], limits[1])

			if k < 1:
				f, lgf = ffg(x, -1, -1)
				eta = min(2 * f / norm(lgf)**2, 10.)
			elif k >= 2:
				# Compute change in solution and gradient
				lx += x
				lg += cg

				nlx = norm(lx)**2
				xdg = abs(np.dot(lx, lg))

				if xdg < sys.float_info.epsilon * nlx:
					# Terminate if step size blows up
					if self.isRoot:
						print 'TERMINATE: epoch', k, 'step size breakdown'
					break

				eta = nlx / xdg / updates

				# Smooth the step (use logs to avoid overflow)
				lkp = np.log(k + 1.0)
				lck = ((k - 2) * lck + np.log(eta) + lkp) / (k - 1.0)
				eta = np.exp(lck - lkp)

			if self.isRoot:
				print 'Epoch', k,
				if rgwt: print 'regularizer weight', rgwt['weight'],
				print 'gradient descent step', eta

			# Copy negative of last solution and gradient
			lx = -x
			lg = -cg

			# Clear gradient for next iteration
			cg[:] = 0.

			# Build a callback to write per-iteration results, if desired
			if self.isRoot:
				cb = self.callbackgen(partial_output, s, k, mfilter)
			else: cb = None

			for t in range(updates):
				# Compute the sampled cost functional and its gradient
				f, lgf = ffg(x, k, t)

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
				cg = beta * lgf + (1 - beta) * cg

			if converged:
				if self.isRoot: print 'TERMINATE: Convergence achieved'
				break

			# Adjust the regularization weight as appropriate
			if ('scale' in rgwt and not (k + 1) % rgwt['every']
					and rgwt['weight'] > rgwt['min']):
				rgwt['weight'] *= rgwt['scale']

		# Update the image
		sp = s.perturb(x)
		# Apply a desired filter
		if mfilter: sp = median_filter(sp, size=mfilter)

		return sp


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

	try:
		# Read straight-ray tomography options, if provided
		sropts = config.get(tsec, 'straight', default={ },
					mapper=dict, checkmap=False)
	except Exception as e: _throw('Invalid optional map "straight"', e)

	if not sropts:
		# Read bent-ray options if straight-ray tomography isn't desired
		try: bropts = config.get(tsec, 'bent', mapper=dict, checkmap=False)
		except Exception as e: _throw('Configuration must specify "bent" map', e)
	else: bropts = { }

	# Read parameters for optional median filter
	try: mfilter = config.get(tsec, 'mfilter', mapper=int, default=None)
	except Exception as e: _throw('Invalid optional mfilter', e)

	try: limits = config.getlist(tsec, 'limits', mapper=float, default=None)
	except Exception as e: _throw('Invalid optional limits', e)

	try: fresnel = config.get(tsec, 'fresnel', mapper=float, default=None)
	except Exception as e: _throw('Invalid optional fresnel', e)

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

	if not sropts:
		# Build the cost calculator
		cshare = BentRayTracer(elements, atimes, tracer, fresnel,
					slowdef, linear, MPI.COMM_WORLD)

		# Compute random updates to the image; SGD can use value limits
		ns = cshare.sgd(slw, mfilter=mfilter, limits=limits, **bropts)
	else:
		# Build the straight-ray tracer
		cshare = StraightRayTracer(elements, atimes,
				tracer.box, fresnel, MPI.COMM_WORLD)
		# Pass configured options to the solver
		ns = cshare.lsmr(slw, **sropts)
		if mfilter: ns = median_filter(ns, size=mfilter)

	if not rank:
		np.save(output, ns.astype(np.float32))
		print 'Finished'
