#!/usr/bin/env python

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

'''
Perform an arrival-time-tomographic udpate of sound speeds.
'''

import sys, os

import functools

import numpy as np
from numpy.linalg import norm

import scipy.ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, norm as snorm

from skimage.restoration import denoise_tv_chambolle

from math import fsum

from random import sample

from time import time

from itertools import count, repeat

from mpi4py import MPI

from pycwp.cytools.regularize import epr, totvar, tikhonov
from pycwp.iterative import lsmr
from pycwp import filter as pcwfilter

from habis.pathtracer import PathTracer, TraceError

from habis.habiconf import (HabisConfigParser, 
				HabisConfigError, matchfiles, watchConfigErrors)
from habis.formats import loadmatlist as ldmats, savez_keymat, loadkeymat
from habis.mpdfile import flocshare, getatimes
from habis.mpfilter import parfilter
from habis.slowness import Slowness, MaskedSlowness, PiecewiseSlowness


class TomographyTask(object):
	'''
	A root class to encapsulate a single MPI rank that participates in
	tomographic reconstructions. Each instances takes responsibility for a
	set of arrival-time measurements provided to it.
	'''
	def __init__(self, elements, atimes, tracer, comm=MPI.COMM_WORLD):
		'''
		Create a worker that collaborates with other ranks in the given
		MPI communicator comm to compute tomographic images. Elements
		have coordinates given by elements[i] for some integer index i,
		and first-arrival times from transmitter t ot receiver r are
		given by atimes[t,r].

		Path integrals (either bent- or straight-ray, with or without
		compensation) will be evaluated with the PathTracer tracer.

		Any arrival times corresponding to a (t, r) pair for which
		elements[t] or elements[r] is not defined, or for which t == r,
		will be discarded.
		'''
		# Record the tracer
		self.tracer = tracer

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


	def save(self, template, s, epoch, filt):
		'''
		Store, in a file whose name is produced by invoking

			template.format(epoch=epoch),

		the slowness s as a 3-D Numpy array. If filt is not None, it
		must be a callable that accepts as its single argument the
		array s. In this case, filt(s) will be called prior to writing
		the output.

		Only the root process will do the writing.
		'''
		if filt: s = filt(s)

		if self.isRoot:
			fname = template.format(epoch=epoch)
			np.save(fname, s.astype(np.float32))


class CSRTomographyTask(TomographyTask):
	'''
	A subclass of TomographyTask to perform compensated straight-ray path
	tracing.
	'''
	def __init__(self, elements, atimes, tracer, comm=MPI.COMM_WORLD):
		'''
		Create a worker for compensated straight-ray tomography on the
		element pairs present in atimes.
		'''
		# Initialize the data and communicator
		super().__init__(elements, atimes, tracer, comm)

		# Figure out the grid dimensions
		ncell = tracer.box.ncell

		# Build the path-length matrix and RHS vector
		indptr = [ 0 ]
		indices = [ ]
		data = [ ]
		self.trpairs = [ ]

		for i, ((t, r), v) in enumerate(sorted(self.atimes.items())):
			path = self.elements[t], self.elements[r]
			plens = tracer.pathmap(path)
			for cidx, l in plens.items():
				data.append(l)
				indices.append(np.ravel_multi_index(cidx, ncell))
			indptr.append(len(indices))
			self.trpairs.append((t, r))

		mshape = len(self.trpairs), np.prod(ncell)
		self.pathmat = csr_matrix((data, indices, indptr),
						shape=mshape, dtype=np.float64)

	def comptimes(self, s, tmin=0.):
		'''
		Compute compensated and straight-ray arrival times for all
		transmit-receive pairs in self.atimes through the medium
		represented by slowness s, a 3-D Numpy array.

		The compensated time adjusts the arrival-time with a factor
		that accounts for deviation of the wavefront normal from the
		straight path. Times are evalauted with self.tracer.trace in
		'straight' mode.

		The argument tmin should be a float such that any path with an
		uncompensated straight-ray arrival time Ts and a compensated
		arrival time Tc that fails to satisfy Ts >= Tc >= tmin * Ts
		will be excluded from the output.

		The return value is a map from transmit-receive index pairs to
		pairs of compensated and uncompensated straight-ray arrival
		times. If a path integral fails for some transmit-receive pair,
		it will not be included in the map.
		'''
		tmin = float(tmin)
		tracer = self.tracer

		# Update the slowness image for the tracer
		tracer.set_slowness(s)

		# Compute the corrective factor for each (t, r) pair
		ivals = { }
		for t, r in self.trpairs:
			path = self.elements[t], self.elements[r]

			try:
				# Try to find the compensated times
				tc, ts = tracer.compensated_trace(path, intonly=True)
				if tc is None or not ts >= tc >= tmin * ts:
					raise ValueError('Compensated time out of range')
			except (ValueError, TraceError): continue

			# Record the times and compensate the RHS
			ivals[t,r] = (tc, ts)

		return ivals

	def lsmr(self, s, epochs=1, coleq=False, tmin=0., chambolle=None,
			postfilter=None, partial_output=None,
			lsmropts={}, omega=1., bent_fallback=False,
			mindiff=False, save_pathmat=None, save_times=None):
		'''
		For each of epochs rounds, compute, using LSMR, a slowness
		image that satisfies the straight-ray arrival-time equations
		implicit in this CSRTomographyTask instance. The solution is
		represented as a perturbation to the slowness s, an instance of
		habis.slowness.Slowness or its descendants, defined on the grid
		self.tracer.box.

		If coleq is True, the columns of each path-length operator will
		be scaled so that they all have unity norm. The value of coleq
		can also be a float that specifies the minimum allowable column
		norm (as a fraction of the maximum norm) to avoid excessive
		scaling of very weak columns.

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

		Within each epoch, an update to the slowness image is computed
		based on the difference between the compensated arrival time
		produced by self.comptimes(s, tmin, bent_fallback) and the
		actual arrival times in self.atimes.

		When mindiff is True, compensated arrival times will be replaced
		by straight-ray times whenever the straight-ray time is closer
		to the measured data for the path.

		The parameter omega should be a float used to damp updates at
		the end of each epoch. In other words, if ds is the update
		computed in an epoch, the slowness s at the end of the epoch
		will be updated according to

			s <- s + omega * ds.

		If chambolle is not None, it should be the "weight" parameter
		to the function skimage.restoration.denoise_tv_chambolle. In
		this case, the denoising filter will be applied to the slowness
		image after each epoch. Alternatively, chambolle can be a list
		of weights, in which case it behaves like the 'maxiter'
		argument of lsmropts.

		After each epoch, if partial_output is True, a partial solution
		will be saved by calling

		  self.save(partial_output, s, epoch, postfilter).

		The return value will be the final, perturbed solution. If
		postfilter is True, the solution s will be processed as
		postfilter(s) before it is returned.

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

		ncell = self.tracer.box.ncell

		# Set a default for Boolean coleq
		if coleq is True: coleq = 1e-3
		elif coleq: coleq = float(coleq)

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

		msgfmt = ('Epoch {0} RMSE {1:0.6g} dsol {2:0.6g} '
				'dct {3:0.6g} dst {4:0.6g} paths {5}')
		epoch, sol, ltimes = 0, 0, { }
		ns = s.perturb(sol)
		while True:
			# Adjust RHS times with straight-ray compensation
			tv = self.comptimes(ns, tmin, bent_fallback)
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
					tv = dict(kp for v in tv for kp in v.items())
					savez_keymat(rname, tv)

			rkeys = [ ]
			rhs = [ ]
			for i, (t, r) in enumerate(self.trpairs):
				try:
					tc, ts = ltimes[t,r]
					ta = self.atimes[t,r]
				except KeyError: continue

				if mindiff: tc = min([tc, ts], key=lambda x: abs(x - ta))

				rhs.append(ta - tc)
				rkeys.append(i)

			# Separate the RHS into row keys and values
			rhs = np.array(rhs)

			lpmat = pathmat[rkeys,:]

			if coleq:
				# Compute norms of columns of global matrix
				colscale = snorm(lpmat, axis=0)**2
				self.comm.Allreduce(MPI.IN_PLACE, colscale, op=MPI.SUM)
				np.sqrt(colscale, colscale)

				# Clip normalization factors to avoid blow-up
				mxnrm = np.max(colscale)
				np.clip(colscale, coleq * mxnrm, mxnrm, colscale)
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

			# Use the right maxiter value for this epoch
			results = lsmr(A, rhs, unorm=unorm, vnorm=norm,
					maxiter=next(itercounts), **lsmropts)

			ds = results[0]
			if colscale is not None: ds /= colscale

			cmwt = next(chamwts)
			if cmwt:
				ds = denoise_tv_chambolle(s.unflatten(ds), cmwt)
				ds = s.flatten(ds)

			if self.isRoot:
				# Compute relative change in solution
				dsnrm = norm(ds) / norm(sol + ds)

				ctnrm = np.sqrt(tdiffs[0] / tdiffs[2])
				stnrm = np.sqrt(tdiffs[1] / tdiffs[3])
				print(msgfmt.format(epoch, terr, dsnrm, ctnrm, stnrm, tn))

			# Update the solution
			sol = sol + omega * ds
			ns = s.perturb(sol)

			if partial_output:
				self.save(partial_output, ns, epoch, postfilter)

			epoch += 1
			if epoch > epochs: break

		if postfilter: ns = postfilter(ns)
		return ns


class SGDTomographyTask(TomographyTask):
	'''
	A subclass of TomographyTask to perform bent-ray path tracing.
	'''
	def __init__(self, elements, atimes,
			tracer, comm=MPI.COMM_WORLD, hitmaps=False):
		'''
		Create a worker to do path tracing, using the provided tracer,
		on the element pairs present in atimes.

		If hitmaps is True, every call to self.evaluate will save a
		copy of the per-voxel hit count and hit density maps in the
		property self.hitmaps. Otherwise, the "hitmaps" property will
		not exist. See the documentation for self.evaluate for more
		information.
		'''
		# Initialize the data and communicator
		super().__init__(elements, atimes, tracer, comm)

		# Save most recent hit map, if desired
		self.hitmaps = None
		self.save_hitmaps = bool(hitmaps)

	def evaluate(self, s, nm, maxerr=None):
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

		If maxerr is not None or 0, it should be a positive float that
		specifies the maximum permissible absolute error allowed for
		any individual path trace. In this case, any measurement pair
		(t,r) that satisfies
		
		  abs(Ls[t,r] - self.atimes[t,r]) > maxerr,

		where Ls[t,r] is the arrival time predicted by a path trace
		from point t to point r, will be treated as if the trace failed
		and excluded from C(s), nr, and grad(C)(s). If maxerr is None
		or 0, this test is not performed.

		The sample consists of nm transmit-receive pairs selected from
		self.atimes with equal probability. If nm >= len(self.atimes)
		or nm < 1, the entire cost functional and gradient will be
		computed. The local sample of the functions is accumulated with
		shares from other MPI ranks in the communicator self.comm using
		reduction operators.

		If self.save_hitmaps is True, self.hitmaps will be updated with
		the hit count and density in a Numpy array of s.shape + (2,),
		where the counts are stored in [:,:,:,0] and the density is
		stored in [:,:,:,1].
		'''
		rank, size = self.comm.rank, self.comm.size
		tracer = self.tracer

		tracer.set_slowness(s)

		# Compute the random sample of measurements
		if nm < 1 or nm > len(self.atimes): nm = len(self.atimes)
		trset = sample(list(self.atimes.keys()), nm)

		# Accumulate the local cost function and gradient
		f = []

		gshape = tracer.box.ncell
		gf = np.zeros(gshape, dtype=np.float64, order='C')

		nrows, nskip = 0, 0

		if self.save_hitmaps:
			hmshape = tracer.get_slowness().shape + (2,)
			hitmaps = np.zeros(hmshape, dtype=np.float64)

		# Make sure that the maximum error makes sense
		maxerr = float(maxerr or 0)

		# Compute contributions for each source-receiver pair
		for t, r in trset:
			path = self.elements[t], self.elements[r]
			atdata = self.atimes[t,r]

			try:
				plens, pint = tracer.trace(path)
				if not plens or pint <= 0:
					raise ValueError
				elif maxerr and abs(pint - atdata) > maxerr:
					raise ValueError

			except (ValueError, TraceError):
				# Skip invalid or empty paths
				nskip += 1
				continue

			nrows += 1
			# Calculate error in model arrival time
			err = pint - atdata
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
			if self.hitmaps is None: self.hitmaps = hitmaps
			else: self.hitmaps += hitmaps

		return f, nrows, gf

	def sgd(self, s, nmeas, epochs, updates, beta=0.5, tol=1e-6,
			maxstep=None, regularizer=None, maxerr=None,
			postfilter=None, partial_output=None):
		'''
		Iteratively compute a slowness image by minimizing the cost
		functional represented in this SGDTomographyTask instance and
		using the solution to update the given slowness model s (an
		instance of habis.slowness.Slowness or its descendants) defined
		over the grid defined in self.tracer.box.

		The Stochastic Gradient Descent, Barzilai-Borwein (SGB-BB)
		method of Tan, et al. (2016) is used to compute the image with
		a convergence tolerance of tol. The method continues for at
		most 'epochs' epochs, with a total of 'updates' stochastic
		descent steps per epoch. A single stochastic descent is made by
		sampling the global cost functional (mean-squared arrival-time
		error) using nmeas measurements per MPI rank.

		The descent step is selecting using a stochastic
		Barzilai-Borwein (BB) scheme. The first two epochs will each
		use a fixed step size. Later updates rely on approximations to
		the gradient in previous epochs. The approximate gradient at
		epoch k is defined recursively over t updates as

		  g_{k,t} = beta * grad(f_t)(x_t) + (1 - beta) g_{k,t-1},

		where g_{k,0} == 0, f_t is the t-th sampled cost functional for
		the epoch and x_t is the solution at update t.

		The "maxerr" argument is passed to self.evaluate.

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

		If maxstep is not None, the BB step size will be clipped above
		the value of maxstep. With no maxstep, the BB step size will
		not be clipped.

		If postfilter is not None, the value postfilter(s) will be
		returned in place of the final solution s.

		If partial_output is not None, the current solution s will be
		saved after every epoch by calling

			self.save(partial_output, s, epoch, postfilter).
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

			See the SGDTomographyTracer documentation for the
			general (unregularized) form of the cost functional.
			'''
			# Track the run time
			stime = time()

			# Compute the perturbed slowness into sp
			sp = s.perturb(x)

			# Compute the stochastic cost and gradient
			f, nrows, gf = self.evaluate(sp, nmeas, maxerr)
			lgf = s.flatten(gf)

			if nrows:
				# Scale cost (and gradient) to mean-squared error
				f /= nrows
				lgf /= nrows

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
				print(f'At epoch {epoch} update {update} '
						f'cost {f:#0.4g} RMSE {rmserr:#0.4g} '
							f'(time: {stime:0.2f} sec)')
				if nrows != maxrows:
					print('  Untraceable paths: '
						f'{maxrows - nrows}/{maxrows}')

			return f, lgf

		# Step smoothing coefficient
		lck = 0.0

		# For convergence testing
		maxcost = 0.0
		converged = False

		if maxstep is not None:
			maxstep = float(maxstep)
			if maxstep <= 0:
				raise ValueError('Value of "maxstep" must be positive')

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
						print(f'TERMINATE: epoch {k} step size breakdown')
					break

				eta = nlx / xdg / updates

				# Smooth the step (use logs to avoid overflow)
				lkp = np.log(k + 1.0)
				lck = ((k - 2) * lck + np.log(eta) + lkp) / (k - 1.0)
				eta = np.exp(lck - lkp)

			# Clip the step size, if desired
			if maxstep: eta = min(eta, maxstep)

			if self.isRoot:
				print('Epoch', k, end=' ')
				if rgwt: print(f'regularizer weight {rgwt["weight"]:#0.6g}', end=' ')
				print('gradient descent step', eta)

			# Copy negative of last solution and gradient
			lx = -x
			lg = -cg

			# Clear gradient for next iteration
			cg[:] = 0.

			for t in range(updates):
				# Compute the sampled cost functional and its gradient
				f, lgf = ffg(x, k, t)

				# Adjust the solution against the gradient
				x -= eta * lgf

				# Check for convergence
				maxcost = max(f, maxcost)
				if f < tol * maxcost:
					converged = True
					break

				# Update the average gradient
				cg = beta * lgf + (1 - beta) * cg

			# Store the partial update if desired
			if partial_output:
				self.save(partial_output, s.perturb(x), k, postfilter)

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
		if postfilter: sp = postfilter(sp)

		return sp


def usage(progname=None, retcode=1):
	if not progname: progname = os.path.basename(sys.argv[0])
	print(f'USAGE: {progname} <configuration>', file=sys.stderr)
	sys.exit(int(retcode))


if __name__ == "__main__":
	if len(sys.argv) != 2: usage()

	try:
		config = HabisConfigParser(sys.argv[1])
	except Exception as e:
		err = 'Unable to load configuration file %s' % sys.argv[1]
		raise HabisConfigError.fromException(err, e)

	rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size

	# Savitzky-Golay coefficients are needed repeatedly; memoize for efficiency
	pcwfilter.savgol = functools.lru_cache(32)(pcwfilter.savgol)

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
	with watchConfigErrors('slowness', tsec):
		s = config.get(tsec, 'slowness')

	# Determine whether piecewise-constant slowness models are desired
	with watchConfigErrors('piecewise', tsec):
		piecewise = config.get(tsec, 'piecewise', mapper=bool, default=False)

	with watchConfigErrors('slowmask', tsec):
		# Load a slowness mask, if one is specified
		mask = config.get(tsec, 'slowmask', default=None)

	# Read required final output
	with watchConfigErrors('output', tsec):
		output = config.get(tsec, 'output')

	with watchConfigErrors('elements', tsec):
		# Read element locations
		efiles = matchfiles(config.getlist(tsec, 'elements'))
		elements = ldmats(efiles, nkeys=1)

	with watchConfigErrors('hitmaps', tsec):
		hitmaps = config.getlist(tsec, 'hitmaps', default=None)

	# Load default background slowness
	with watchConfigErrors('slowdef', tsec):
		slowdef = config.get(tsec, 'slowdef', mapper=float, default=None)

	# Determine interpolation mode
	with watchConfigErrors('linear', tsec):
		linear = config.get(tsec, 'linear', mapper=bool, default=True)

	with watchConfigErrors('tfilter', tsec):
		tfilter = config.get(tsec, 'tfilter', default=None)

	with watchConfigErrors('csr', tsec):
		# Read straight-ray tomography options, if provided
		csropts = config.get(tsec, 'csr', default={ },
					mapper=dict, checkmap=False)

	if not csropts:
		# Read bent-ray options if straight-ray tomography isn't desired
		with watchConfigErrors('sgd', tsec):	
			sgdopts = config.get(tsec, 'sgd', mapper=dict, checkmap=False)
	else: sgdopts = { }

	# Read parameters for optional median filter
	with watchConfigErrors('mfilter', tsec):
		mfilter = config.get(tsec, 'mfilter', mapper=int, default=None)

	with watchConfigErrors('maskoutliers', tsec):
		mask_outliers = config.getlist(tsec, 'maskoutliers', default=False)

	if not rank and mask_outliers: print('Will exclude outlier arrival times')

	with watchConfigErrors('partial_output', tsec):
		partial_output = config.get(tsec, 'partial_output', default=None)

	with watchConfigErrors('vclip', tsec):
		# Pull a range of valid sound speeds for clipping
		vclip = config.getlist(tsec, 'vclip', mapper=float, default=None)
		if vclip:
			if len(vclip) != 2:
				raise ValueError('Range must specify two elements')
			elif not rank: print('Will limit average path speeds to', vclip)

	with watchConfigErrors('timefile', tsec):
		# Look for local files and determine local share
		tfiles = matchfiles(config.getlist(tsec, 'timefile'), forcematch=False)
		# Determine the local shares of every file
		tfiles = flocshare(tfiles, MPI.COMM_WORLD)
		# Pull out local share of locally available arrival times
		atimes = { }
		for tf, (st, ln) in tfiles.items():
			atimes.update(getatimes(tf, elements, 0, False,
						vclip, mask_outliers, st, ln))

	with watchConfigErrors('exclusions', tsec):
		# Try to load a list of arrival times to exclude
		efiles = matchfiles(config.getlist(tsec, 'exclusions'), forcematch=False)
		exclusions = { (t,r) for f in efiles
				for r, tl in loadkeymat(f).items() for t in tl }
		if exclusions:
			if not rank:
				print(f'{len(exclusions)} measurement pairs marked for exclusion')
			goodpairs = set(atimes).difference(exclusions)
			atimes = { k: atimes[k] for k in goodpairs }

	# Convert the scalar slowness or file name into a matrix
	try: s = float(s)
	except (ValueError, TypeError): s = np.load(s).astype(np.float64)
	else: s = s * np.ones(bx.ncell, dtype=np.float64)

	if piecewise:
		if mask is None:
			raise ValueError('Slowness mask is required in piecewise mode')
		mask = np.load(mask)
		slw = PiecewiseSlowness(mask, s)
		if not rank: print(f'Using piecewise slowness')
	else:
		if mask is not None:
			mask = np.load(mask).astype(bool)
			slw = MaskedSlowness(s, mask)
			if not rank: print(f'Using masked slowness')
		else: slw = Slowness(s)

	# Collect the total number of arrival-time measurements
	timecount = MPI.COMM_WORLD.reduce(len(atimes), op=MPI.SUM)
	if not rank: print(f'{timecount:,d} total measurements, {slw.nnz:,d} unknowns')

	if tracer.prefilter and not rank:
		print(f'Will use {tracer.prefilter} as a pre-integral filter')

	comm = MPI.COMM_WORLD

	if mfilter:
		filt = parfilter('median_filter', comm)
		def postfilter(s):
			return filt(s, size=mfilter)
	else: postfilter = None

	if not csropts:
		# Build the cost calculator
		cshare = SGDTomographyTask(elements, atimes, tracer, comm, hitmaps)

		# Compute random updates to the image
		ns = cshare.sgd(slw, postfilter=postfilter,
				partial_output=partial_output, **sgdopts)

		# Grab the hitmaps, if they were saved
		hmaps = cshare.hitmaps
	else:
		# Build the straight-ray tracer
		cshare = CSRTomographyTask(elements, atimes, tracer, comm)

		if not rank:
			print('Finished building straight-ray tracer object')

		if hitmaps:
			# Save the hitmaps
			hmaps = np.zeros(tracer.box.ncell + (2,), dtype=np.float64)

			for k, v in cshare.pathmat.todok().items():
				l,m,n = np.unravel_index(k[1], tracer.box.ncell)
				hmaps[l,m,n,0] += 1.
				hmaps[l,m,n,1] += v

			MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, hmaps, op=MPI.SUM)
		else: hmaps = None

		# Find the solution
		ns = cshare.lsmr(slw, postfilter=postfilter,
				partial_output=partial_output, **csropts)

	if not rank:
		np.save(output, ns.astype(np.float32))

		if hmaps is not None:
			print('Will save hit maps')
			np.save(hitmaps[0], hmaps[:,:,:,0])
			try: np.save(hitmaps[1], hmaps[:,:,:,1])
			except IndexError: pass

		print('Finished')
