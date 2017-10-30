#!/usr/bin/env python

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

'''
Perform an arrival-time-tomographic udpate of sound speeds.
'''

import sys, os

import numpy as np
from numpy.linalg import norm

import scipy.ndimage
from scipy.ndimage import median_filter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, norm as snorm

from skimage.restoration import denoise_tv_chambolle

from math import fsum

from random import sample

from time import time

from itertools import count, repeat

from mpi4py import MPI

from pycwp.cytools.interpolator import HermiteInterpolator3D, LinearInterpolator3D, CubicInterpolator3D
from pycwp.cytools.regularize import epr, totvar, tikhonov
from pycwp.iterative import lsmr
from pycwp.cytools.boxer import Segment3D

from habis.pathtracer import PathTracer, WavefrontNormalIntegrator, TraceError

from habis.habiconf import HabisConfigParser, HabisConfigError, matchfiles
from habis.formats import loadmatlist as ldmats, savez_keymat
from habis.mpdfile import flocshare, getatimes
from habis.slowness import Slowness, MaskedSlowness, PiecewiseSlowness


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

		for (t, r), v in atimes.items():
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
	def __init__(self, elements, atimes, box, fresnel=None,
			slowdef=None, linear=True, comm=MPI.COMM_WORLD):
		'''
		Create a worker for straight-ray tomography on the element
		pairs present in atimes. The imaging grid is represented by the
		Box3D box.

		If Fresnel is not None, it should be a positive floating-point
		value. In this case, rays are represented as ellipsoids that
		embody the first Fresnel zone for a principal wavelength of
		fresnel (in units that match the units of box.cell).

		For time compensation (if applied), slowness will be
		interpolated with LinearInterpolato3D if linear is True
		(CubicInterpolator3D otherwise). The interpolator will inherit
		a default (out-of-bounds) slowness slowdef.
		'''
		# Initialize the data and communicator
		super(StraightRayTracer, self).__init__(elements, atimes, comm)

		# Grab a reference to the box
		self.box = box
		ncell = self.box.ncell

		# Identify the right interpolator for multi-round imaging
		self.interpolator = (LinearInterpolator3D if linear
						else CubicInterpolator3D)
		self.slowdef = slowdef

		# Build the path-length matrix and RHS vector
		indptr = [ 0 ]
		indices = [ ]
		data = [ ]
		self.trpairs = [ ]
		self.rhs = { }

		for i, ((t, r), v) in enumerate(sorted(self.atimes.items())):
			# March straight through the grid
			seg = Segment3D(self.elements[t], self.elements[r])
			# Compute the total segment length
			if not fresnel:
				# Build a simple row of the path-length matrix
				path = self.box.raymarcher(seg)
				for cidx, (s, e) in path.items():
					data.append(seg.length * (e - s))
					indices.append(np.ravel_multi_index(cidx, ncell))
			else:
				# Build a row based on the first Fresnel zone
				hits = self.box.fresnel(seg, fresnel)
				htot = fsum(iter(hits.values()))
				for cidx, ll in hits.items():
					data.append(seg.length * ll / htot)
					indices.append(np.ravel_multi_index(cidx, ncell))
			indptr.append(len(indices))
			self.rhs[i] = v
			self.trpairs.append((t, r))

		mshape = len(self.trpairs), np.prod(ncell)
		self.pathmat = csr_matrix((data, indices, indptr),
						shape=mshape, dtype=np.float64)

	def timeadjust(self, si, tmin=0., omega=1., **kwargs):
		'''
		Adjust the arrival times in self.rhs with a corrective factor
		that tracks wavefront normal directions using the method of
		habis.pathtracer.WavefrontNormalIntegrator.

		The argument si should be an Interpolator3D instance that
		represents a slowness model used to adjust the times.

		The argument tmin should be a float such that any path with an
		actual arrival time T and a compensated arrival time Tc that
		fails to satisfy Tc >= tmin * T will be excluded from self.rhs.

		The argument omega should be a float used to damp time
		adjustments. For a given path with measured arrival time T in
		self.atimes, compensated arrival time Tc and straight-ray
		arrival time Ts (each in the presence of the model si), the
		adjusted time for the path is

			Ta = Ts + omega * (T - Tc).

		The remaining keyword arguments kwargs are passed to the method
		WavefrontNormalIntegrator.pathint to control the corrective
		terms. The keyword "h" is forbidden.

		The return value is a map from transmit-receive index pairs to
		pairs of compensated and uncompensated straight-ray arrival
		times. If a path integral fails for some transmit-receive pair,
		or if the compensated integral does not fall between the
		straight-ray integral and the true arrival time, it will not be
		included in the map.
		'''
		# Build the integrator
		integrator = WavefrontNormalIntegrator(si)

		tmin = float(tmin)

		# Compute the corrective factor for each (t, r) pair
		cell = self.box.cell
		self.rhs = { }
		ivals = { }
		for i, (t, r) in enumerate(self.trpairs):
			vt = self.atimes[t,r]
			tx = bx.cart2cell(*self.elements[t])
			rx = bx.cart2cell(*self.elements[r])

			try:
				# Try to find the compensated straight-ray time
				tc, ts = integrator.pathint(tx, rx, h=cell, **kwargs)
				# Check the time for sanity
				if not ts >= tc >= tmin * vt:
					raise ValueError('Compensated time out of range')
			except ValueError: continue

			# Record the times and compensate the RHS
			ivals[t,r] = (tc, ts)
			self.rhs[i] = ts + omega * (vt - tc)

		return ivals

	@staticmethod
	def save(template, s, epoch, mfilter):
		'''
		Store, in a file whose name is produced by invoking

			template.format(epoch=epoch),

		the slowness s as a 3-D Numpy array. If mfilter is True, it
		should be a value passed as the "size" argument to
		scipy.ndimage.median_filter that will smooth the slowness
		before output.
		'''
		fname = template.format(epoch=epoch)

		if mfilter: s = median_filter(s, size=mfilter)
		np.save(fname, s.astype(np.float32))

	def lsmr(self, s, epochs=1, coleq=False, pathopts={}, chambolle=None,
			tfilter=None, mfilter=None, partial_output=None,
			lsmropts={}, save_pathmat=None, save_times=None):
		'''
		For each of epochs rounds, compute, using LSMR, a slowness
		image that satisfies the straight-ray arrival-time equations
		implicit in this StraightRayTracer instance. The solution is
		represented as a perturbation to the slowness s, an instance of
		habis.slowness.Slowness or its descendants, defined on the grid
		self.box.

		If coleq is True, the columns of each path-length operator will
		be scaled so that they all have unity norm.

		The LSMR implementation uses pycwp.iterative.lsmr to support
		arrival times distributed across multiple MPI tasks in
		self.comm. The keyword arguments lsmropts will be passed to
		lsmr to customize the solution process. Forbidden keyword
		arguments are "A", "b", "unorm" and "vnorm".

		If lsmropts contains a 'maxiter' keyword, it can be a single
		integer or a list of integers. If the value is a list, it
		provides the maximum number of LSMR iterations for each epoch
		in sequence. If the total number of epochs exceeds the number
		of values in the list, the final value will be repeated as
		necessary.

		Between subsequent epochs, the arrival times will be updated
		according to self.timeadjust(si, **pathopts) to incorporate
		straight-ray compensations. The argument si is an
		Interpolator3D representation of s (linear if self.linear is
		True, cubic otherwise).

		If tfilter is not None, it should be a tuple of the form
		(name, size), where name is a string used to select an image
		filter as scipy.ndimage.<name>_filter and size is the second
		argument to the filter (the image to filter is the first
		argument). The chosen filter will be applied to the slowness
		image when producing the interpolator used by self.timeadjust.
		The filter will not influence the evolution of images between
		epochs. If tfilter is None, no filtering is attempted.

		If chambolle is not None, it should be the "weight" parameter
		to the function skimage.restoration.denoise_tv_chambolle. In
		this case, the denoising filter will be applied to the slowness
		image after each epoch. Alternatively, chambolle can be a list
		of weights, in which case it behaves like the 'maxiter'
		argument of lsmropts.

		After each epoch, if partial_output is True, a partial solution
		will be saved by calling

		  self.lsmr(partial_output, s, epoch, mfilter).

		The return value will be the final, perturbed solution. If
		mfilter is True, the solution will be processed with
		scipy.ndimage.median_filter before it is written.

		If save_pathmat is not None, it should be a string template
		which will be formatted with

		  pname = save_pathmat.format(rank=self.comm.rank)

		and used as a file name in which the reduced path-length matrix
		will be stored as a COO matrix in an NPZ file with keys data,
		row and col, corresponding to the attributes of the COO matrix.
		An additional key, 'trpairs', records the transmit-receive
		pairs that correspond to each row in the local path matrix.

		If save_times is not None, it should be a string template which
		will be formatted as

		  rname = save_times.format(epoch=epoch)

		After each epoch, an array of compensated and uncompensated
		straight-ray path integrals will be stored in a keymat file
		with name rname. All times will be coalesced onto the root rank
		for output.
		'''
		if not self.isRoot:
			# Make sure non-root tasks are always silent
			lsmropts['show'] = False

		if 'calc_var' in lsmropts:
			raise TypeError('Argument "calc_var" is forbidden')

		ncell = self.box.ncell

		if tfilter:
			tfname = tfilter[0].lower() + '_filter'
			try: timefilt = getattr(scipy.ndimage, tfname)
			except AttributeError: timefilt = None
			tfwidth = tfilter[1]
		else: timefilt, tfwidth = None, None

		# Composite slowness transform and path-length operator as CSR
		pathmat = (self.pathmat @ s.tosparse()).tocsr()

		if save_pathmat:
			# Save the path-length matrix
			pname = save_pathmat.format(rank=self.comm.rank)
			pcoo = pathmat.tocoo()
			np.savez(pname, data=pcoo.data, row=pcoo.row,
					col=pcoo.col, trpairs=self.trpairs)
			del pcoo, pname

		def unorm(u):
			# Synchronize
			self.comm.Barrier()
			# Norm of distributed vectors, to all ranks
			un = norm(u)**2
			return np.sqrt(self.comm.allreduce(un, op=MPI.SUM))

		# Process maxiter argument to allow per-epoch specification
		maxiter = lsmropts.pop('maxiter', None)
		try: maxiter = list(maxiter)
		except TypeError: itercounts = repeat(maxiter)
		else: itercounts = (maxiter[min(i, len(maxiter)-1)] for i in count())

		try: chambolle = list(chambolle)
		except TypeError: chamwts = repeat(chambolle)
		else: chamwts = (chambolle[min(i, len(chambolle)-1)] for i in count())

		msgfmt = 'Epoch %d RMSE %0.6g dsol %0.6g dct %0.6g dst %0.6g paths %d'
		epoch, sol, ltimes = 0, 0, { }
		while True:
			# Separate the RHS into row keys and values
			rkeys, rhs = zip(*self.rhs.items())
			rhs = np.array(rhs)

			lpmat = pathmat[rkeys,:]

			if coleq:
				# Compute norms of columns of global matrix
				colscale = snorm(lpmat, axis=0)**2
				self.comm.Allreduce(MPI.IN_PLACE, colscale, op=MPI.SUM)
				np.sqrt(colscale, colscale)

				# Clip normalization factors to avoid blow-up
				mxnrm = np.max(colscale)
				np.clip(colscale, 1e-3 * mxnrm, mxnrm, colscale)
			else: colscale = None

			# Include column scaling in the matrix-vector product
			def mvp(x):
				if colscale is not None: x = x / colscale
				v = lpmat @ x
				return v

			# Transpose operation requires communications
			def amvp(u):
				# Synchronize
				self.comm.Barrier()
				# Multiple by transposed local share, then flatten
				v = lpmat.T @ u
				if colscale is not None: v /= colscale
				# Accumulate contributions from all ranks
				self.comm.Allreduce(MPI.IN_PLACE, v, op=MPI.SUM)
				return v

			# Build the linear operator representing the path matrix
			A = LinearOperator(shape=lpmat.shape, matvec=mvp,
						rmatvec=amvp, dtype=lpmat.dtype)

			# RHS is arrival times minus unperturbed solution
			rhs -= self.pathmat[rkeys,:] @ s.perturb(sol).ravel('C')

			# Use the right maxiter value for this epoch
			results = lsmr(A, rhs, unorm=unorm, vnorm=norm,
					maxiter=next(itercounts), **lsmropts)

			ds = results[0]
			if colscale is not None: ds /= colscale

			cmwt = next(chamwts)
			if cmwt:
				ds = denoise_tv_chambolle(s.unflatten(ds), cmwt)
				ds = s.flatten(ds)

			# Compute relative change in solution
			dsnrm = norm(ds) / norm(sol + ds)

			# Update the solution
			sol = sol + ds
			ns = s.perturb(sol)

			if partial_output:
				self.save(partial_output, ns, epoch, mfilter)

			if epochs:
				# Filter and interpolate image for time compensation
				if timefilt and tfwidth > 0:
					ns = timefilt(ns, tfwidth)
				nsi = self.interpolator(ns)
				nsi.default = slowdef

				# Adjust RHS times with straight-ray compensation
				tv = self.timeadjust(nsi, **pathopts)
				# Find RMS arrival-time error for compensated model
				terr = fsum((v[0] - self.atimes[k])**2
						for k, v in tv.items())
				tn = self.comm.allreduce(len(tv), op=MPI.SUM)
				terr = self.comm.allreduce(terr, op=MPI.SUM)
				terr = np.sqrt(terr / tn)

				# Compute norms of model times and time changes
				tdiffs = [ ]
				for k in set(tv).intersection(ltimes):
					a, b = tv[k]
					c, d = ltimes[k]
					tdiffs.append([(a-c)**2, (b-d)**2, a**2, b**2])
				# Reduce square norms across all ranks
				if tdiffs: tdiffs = np.sum(tdiffs, axis=0)
				else: tdiffs = np.array([0.]*4, dtype=np.float64)
				self.comm.Allreduce(MPI.IN_PLACE, tdiffs, op=MPI.SUM)
				# Set placeholder values if there is no value
				if tdiffs[2] == 0: tdiffs[0] = tdiffs[2] = 1.
				if tdiffs[3] == 0: tdiffs[1] = tdiffs[3] = 1.
				ltimes = tv

				if save_times:
					tv = self.comm.gather(tv)
					if self.isRoot:
						rname = save_times.format(epoch=epoch)
						tv = dict(kp for v in tv
								for kp in v.items())
						savez_keymat(rname, tv)
			else:
				# Without compensation, error is LSMR residual
				tn = self.comm.allreduce(len(self.rhs), op=MPI.SUM)
				terr = results[3] / np.sqrt(tn)
				tdiffs = [1.] * 4

			if self.isRoot:
				ctnrm = np.sqrt(tdiffs[0] / tdiffs[2])
				stnrm = np.sqrt(tdiffs[1] / tdiffs[3])
				print(msgfmt % (epoch, terr, dsnrm, ctnrm, stnrm, tn))

			epoch += 1
			if epoch > epochs: break

		if mfilter: ns = median_filter(ns, size=mfilter)
		return ns


class BentRayTracer(TomographyTask):
	'''
	A subclass of TomographyTask to perform bent-ray path tracing.
	'''
	def __init__(self, elements, atimes, tracer, fresnel=None,
			slowdef=None, linear=True, comm=MPI.COMM_WORLD, hitmaps=False):
		'''
		Create a worker to do path tracing on the element pairs present
		in atimes. Slowness will be interpolated with
		LinearInterpolator3D if linear is True (HermiteInterpolator3D
		otherwise). The interpolator will inherit a default
		(out-of-bounds) slowness slowdef.

		If hitmaps is True, every call to self.evaluate will save a
		copy of the per-voxel hit count and hit density maps in the
		property self.hitmaps. Otherwise, the "hitmaps" property will
		not exist. See the documentation for self.evaluate for more
		information.
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

		# Save most recent hit map, if desired
		self.save_hitmaps = bool(hitmaps)

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

		If self.save_hitmaps is True, self.hitmaps will be updated with
		the hit count and density in a Numpy array of s.shape + (2,),
		where the counts are stored in [:,:,:,0] and the density is
		stored in [:,:,:,1].
		'''
		rank, size = self.comm.rank, self.comm.size

		# Interpolate the slowness
		si = self.interpolator(s)
		si.default = self.slowdef

		# The Fresnel wavelength parameter (or None)
		fresnel = self.fresnel

		# Compute the random sample of measurements
		if nm < 1 or nm > len(self.atimes): nm = len(self.atimes)
		trset = sample(list(atimes.keys()), nm)

		# Accumulate the local cost function and gradient
		f = []

		gshape = self.tracer.box.ncell
		gf = np.zeros(gshape, dtype=np.float64, order='C')

		nrows, nskip = 0, 0

		if self.save_hitmaps:
			hitmaps = np.zeros(si.shape + (2,), dtype=np.float64)

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
			for c, l in plens.items(): gf[c] += l * err

			if self.save_hitmaps:
				# Update the hit maps
				for c, l in plens.items():
					hitmaps[c + (0,)] += 1.
					hitmaps[c + (1,)] += l

		# Accumulate the cost functional and row count
		f = self.comm.allreduce(0.5 * fsum(f), op=MPI.SUM)
		nrows = self.comm.allreduce(nrows, op=MPI.SUM)
		# Use the lower-level routine for the in-place gradient accumulation
		self.comm.Allreduce(MPI.IN_PLACE, gf, op=MPI.SUM)

		if self.save_hitmaps:
			self.comm.Allreduce(MPI.IN_PLACE, hitmaps, op=MPI.SUM)
			self.hitmaps = hitmaps

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
			regularizer=None, mfilter=None, partial_output=None):
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
				print(msgfmt % (epoch, update, f, rmserr, stime))
				if nrows != maxrows:
					print('%d/%d untraceable paths' % (maxrows - nrows, maxrows))

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
			if k < 1:
				f, lgf = ffg(x, -1, -1)
				eta = min(2 * f / norm(lgf)**2, 10.)
			elif k >= 2:
				# Compute change in solution and gradient
				lx += x
				lg += cg

				nlx = norm(lx)**2
				xdg = abs(lx @ lg)

				if xdg < sys.float_info.epsilon * nlx:
					# Terminate if step size blows up
					if self.isRoot:
						print('TERMINATE: epoch', k, 'step size breakdown')
					break

				eta = nlx / xdg / updates

				# Smooth the step (use logs to avoid overflow)
				lkp = np.log(k + 1.0)
				lck = ((k - 2) * lck + np.log(eta) + lkp) / (k - 1.0)
				eta = np.exp(lck - lkp)

			if self.isRoot:
				print('Epoch', k, end=' ')
				if rgwt: print('regularizer weight', rgwt['weight'], end=' ')
				print('gradient descent step', eta)

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
				if self.isRoot: print('TERMINATE: Convergence achieved')
				break

			# Adjust the regularization weight as appropriate
			if ('scale' in rgwt and not (k + 1) % rgwt['every']
					and rgwt['weight'] > rgwt['min']):
				rgwt['weight'] *= rgwt['scale']
				if rgwt['weight'] < rgwt['min']:
					# Clip small regularizer weights, stop scaling
					rgwt['weight'] = rgwt['min']
					del rgwt['scale']

		# Update the image
		sp = s.perturb(x)
		# Apply a desired filter
		if mfilter: sp = median_filter(sp, size=mfilter)

		return sp


def usage(progname=None, retcode=1):
	if not progname: progname = os.path.basename(sys.argv[0])
	print('USAGE: %s <configuration>' % progname, file=sys.stderr)
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
		print('Solution defined on grid', bx.lo, bx.hi, bx.ncell)

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
	except Exception as e: _throw('Invalid optional slowmask', e)

	# Read required final output
	try: output = config.get(tsec, 'output')
	except Exception as e: _throw('Configuration must specify output', e)

	try:
		# Read element locations
		efiles = matchfiles(config.getlist(tsec, 'elements'))
		elements = ldmats(efiles, nkeys=1)
	except Exception as e: _throw('Configuration must specify elements', e)

	try: hitmaps = config.getlist(tsec, 'hitmaps', default=None)
	except Exception as e: _throw('Invalid optional hitmaps', e)

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

	try: mask_outliers = config.getlist(tsec, 'maskoutliers', default=False)
	except Exception as e: _throw('Invalid optional maskoutliers', e)

	if not rank and mask_outliers: print('Will exclude outlier arrival times')

	try: fresnel = config.get(tsec, 'fresnel', mapper=float, default=None)
	except Exception as e: _throw('Invalid optional fresnel', e)

	try: partial_output = config.get(tsec, 'partial_output', default=None)
	except Exception as e: _throw('Invalid optional partial_output', e)

	try:
		# Pull a range of valid sound speeds for clipping
		vclip = config.getlist(tsec, 'vclip', mapper=float, default=None)
		if vclip:
			if len(vclip) != 2:
				raise ValueError('Range must specify two elements')
			elif not rank: print('Will limit average path speeds to', vclip)
	except Exception as e:
		_throw('Invalid optional vclip', e)

	try:
		# Look for local files and determine local share
		tfiles = matchfiles(config.getlist(tsec, 'timefile'), forcematch=False)
		# Determine the local shares of every file
		tfiles = flocshare(tfiles, MPI.COMM_WORLD)
		# Pull out local share of locally available arrival times
		atimes = { }
		for tf, (st, ln) in tfiles.items():
			atimes.update(getatimes(tf, elements, 0, False,
						vclip, mask_outliers, st, ln))
	except Exception as e: _throw('Configuration must specify valid timefile', e)

	if piecewise:
		if mask is None:
			raise ValueError('Slowness mask is required in piecewise mode')
		mask = np.load(mask)
		slw = PiecewiseSlowness(mask, s)
		if not rank: print('Using piecewise slowness model')
	else:
		# Convert the scalar slowness or file name into a matrix
		try: s = float(s)
		except (ValueError, TypeError): s = np.load(s).astype(np.float64)
		else: s = s * np.ones(bx.ncell, dtype=np.float64)

		if mask is not None:
			mask = np.load(mask).astype(bool)
			slw = MaskedSlowness(s, mask)
		else: slw = Slowness(s)

	# Collect the total number of arrival-time measurements
	timecount = MPI.COMM_WORLD.reduce(len(atimes), op=MPI.SUM)
	if not rank:
		print('Using a total of %d arrival-time measurements' % (timecount,))

	if not sropts:
		# Build the cost calculator
		cshare = BentRayTracer(elements, atimes, tracer, fresnel,
					slowdef, linear, MPI.COMM_WORLD, hitmaps)

		# Compute random updates to the image
		ns = cshare.sgd(slw, mfilter=mfilter,
				partial_output=partial_output, **bropts)

		# Grab the hitmaps, if they were saved
		try: hmaps = cshare.hitmaps
		except AttributeError: hmaps = None
	else:
		# Build the straight-ray tracer
		cshare = StraightRayTracer(elements, atimes, tracer.box,
				fresnel, slowdef, linear, MPI.COMM_WORLD)

		if hitmaps:
			# Save the hitmaps
			hmaps = np.zeros(cshare.box.ncell + (2,), dtype=np.float64)

			for k, v in cshare.pathmat.todok().items():
				l,m,n = np.unravel_index(k[1], cshare.box.ncell)
				hmaps[l,m,n,0] += 1.
				hmaps[l,m,n,1] += v

			MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, hmaps, op=MPI.SUM)
		else: hmaps = None

		# Find the solution
		ns = cshare.lsmr(slw, mfilter=mfilter,
				partial_output=partial_output, **sropts)

	if not rank:
		np.save(output, ns.astype(np.float32))

		if hmaps is not None:
			print('Will save hit maps')
			np.save(hitmaps[0], hmaps[:,:,:,0])
			try: np.save(hitmaps[1], hmaps[:,:,:,1])
			except IndexError: pass

		print('Finished')
