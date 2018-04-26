'''
Classes for modeling slowness on a 3-D voxel grid as a function of some number
of parameters.
'''

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, eye as speye

import re
from itertools import product as iproduct

class Slowness(object):
	'''
	A class representing an unconstrained slowness on a 3-D grid, where
	each parameter represents the slowness at a voxel in row-major order.
	'''
	def __init__(self, s, shape=None):
		'''
		Initialize a Slowness instance with a reference s, which must
		be a rank-3 floating-point array or a scalar.

		If s is a scalar, shape must be a 3-tuple of positive integers.
		If s is a Numpy array, shape must match s.shape.
		'''
		# Make a copy of the array (may be a scalar)
		self._s = np.array(s, dtype=np.float64)

		if shape is None: shape = self._s.shape
		else: shape = tuple(int(s) for s in shape)

		if not shape:
			raise ValueError('Either s must have a shape, or the shape argument is required')

		if len(shape) != 3:
			raise ValueError('Slowness shape must be three-dimensional')
		if any(sv < 1 for sv in shape):
			raise ValueError('Slowness shape must contain positive values')

		if not self._s.shape:
			self._shape = shape
		else:
			self._shape = None
			if shape != self._s.shape:
				raise ValueError('Argument shape, if provided, must agree with s.shape')

	@property
	def shape(self):
		'''The shape of the slowness model'''
		return self._shape or self._s.shape

	@property
	def nnz(self):
		'''The number of parameters in the slowness model'''
		return np.prod(self.shape)


	def perturb(self, x, persist=False):
		'''
		Perturb the reference slowness by adding x, which must either
		be a scalar or have shape compatible with self.shape. The input
		x will be reshaped in row-major order to match the voxel grid.

		If persist is True, the underlying slowness will be updated, so
		that the sequence of calls

			self.perturb(x, persist=True)
			self.perturb(0)

		will yield the same results.
		'''
		# Expand perturbation onto voxel grid, then add reference
		out = self.unflatten(x) + self._s

		# Save perturbation as new reference
		if persist: self._s = np.array(out)

		return out


	def clip(self, x, smin, smax):
		'''
		Clip the perturbation x so that self.perturb(x) clips to the
		range [smin, smax].
		'''
		out = self.perturb(x)
		# Clip the perturbed slowness and subtract the background
		np.clip(out, smin, smax, out)
		out -= self._s
		return out


	def flatten(self, s):
		'''
		Flatten the array s, with shape self.shape, into a 1-D array.
		'''
		# Just unravel in C order
		return np.ravel(s, order='C')


	def unflatten(self, s):
		'''
		Unflatten the array s, which must be a scalar or have a shape
		compatible with (self.nnz,), into an array of shape self.shape.
		'''
		shape = self.shape
		# Make sure non-scalar s has the right shape
		s = np.asarray(s)
		if s.shape:
			out = s.reshape(shape, order='C')
		else:
			out = np.empty(shape, dtype=s.dtype)
			out[:,:,:] = s
		return out


	def tosparse(self):
		'''
		Return a scalar representation of the linear transformation
		represented by this Slowness, which is just an identity.
		'''
		return speye(self.nnz, dtype=np.float64, format='csr')


class MaskedSlowness(Slowness):
	'''
	A class representing an unconstrained slowness on a masked 3-D grid.

	A slowness parameter will exist for every voxel masked True, while
	voxels masked False will remain fixed according to the instantiated
	reference slowness.
	'''
	def __init__(self, s, mask, shape=None):
		'''
		Initialized a MaskedSlowness instance with reference s.

		The arguments "s" and "shape" are interpreted as in the base
		Slowness class. The argument "mask" should be a Boolean
		array (or compatible) with a shape that matches s.shape or
		shape (as appropriate). See the class docstring for more
		information about the mask.
		'''
		# Copy the mask and use its shape by default
		mask = np.array(mask, dtype=bool)
		if shape is None: shape = mask.shape

		# Initialize the background slowness
		super(MaskedSlowness, self).__init__(s, shape)

		# Check agreement of the mask
		self._mask = mask
		if self._mask.shape != self.shape:
			raise ValueError('Array "mask" must have shape %s' % (self.shape,))

		# Figure the number of parameters in the model
		self._nnz = np.sum(self._mask)

	@property
	def nnz(self):
		'''The number of parameters in the slowness model'''
		return self._nnz

	@property
	def mask(self):
		'''A copy of the Boolean voxel mask'''
		return self._mask.copy()


	def clip(self, x, smin, smax):
		'''
		Clip the perturbation x so that self.perturb(x) clips to the
		range [smin, smax] for True-masked voxels.
		'''
		out = np.array(x)
		if not out.shape:
			out = np.empty((self.nnz,), dtype=out.dtype)
			out[:] = x
		# Add the reference to the perturbation and clip
		rs = self._s[self._mask] if self._s.shape else self._s
		out += rs
		np.clip(out, smin, smax, out)
		# Subtract the reference
		out -= rs
		return out


	def flatten(self, s):
		'''
		Flatten the array s, with shape self.shape, into a 1-D array of
		shape (self.nnz,) by pulling only the True-masked values of s.
		'''
		return s[self._mask]


	def unflatten(self, s):
		'''
		Expand the array s, which must be a scalar or have a shape
		(self.nnz,), into a 3-D grid where self.mask is True. Voxels
		masked False will be 0.
		'''
		out = np.zeros(self.shape, dtype=self._s.dtype)
		out[self._mask] = s
		return out


	def tosparse(self):
		'''
		Return a CSR representation of the linear transformation
		represented by this Slowness, which is an M x N matrix
		consisting of an N x N permutation matrix with a collection of
		(M - N) zero rows inserted throughout the matrix. In this case,
		N = self.nnz and M = product(self.mask.shape).
		'''
		# Flatten the mask for operator representations
		mask = self._mask.ravel('C')
		# Nonzero row indices in operator are nonzero indices in flat mask
		rows = np.nonzero(mask)[0]
		# Columns proceed sequentially
		cols = np.arange(len(rows))
		# Nonzero entries are all unity
		data = np.ones((len(rows),), dtype=np.float64)

		# Return the sparse linear operator
		M, N = len(mask), len(cols)
		return csr_matrix((data, (rows, cols)), shape=(M, N))


class PiecewiseSlowness(Slowness):
	'''
	A class representing a piecewise-constant slowness on a 3-D grid, where
	each parameter represents an additive perturbation to a particular
	slowness for a predetermined set of voxels.
	'''
	def __init__(self, voxmap, s):
		'''
		Initialize a PiecewiseSlowness instance. The map voxmap should
		map keys to 3-D array of shape (L, M, N) such that, if

		  keys = sorted(voxmap.keys()),

		a slowness image for a perturbation x is given by

		slowness = s + sum(voxmap[keys[i]] * x[i]
					for i in range(len(x))).

		A special class of keys, starting with 'unconstrained', will
		behave differently. Each nonzero voxel in the image at
		voxmap['unconstrained'] will effectively get its own key,
		allowing each nonzero pixel in the 'unconstrained'
		voxmap to take a distinct value.

		Additional 'unconstrained' keys are allowed to take the form
		'unconstrained_<d>x', where '<d>' is an arbitrary base-10
		integer. Each nonzero voxel in I = voxmap['unconstrained_Dx']
		for an integer D represents a unique value corresponding to a
		cluster of DxDxD ordinary voxels; i.e., each voxel (i,j,k) in I
		corresponds to an effective map M_ijk with

		  M_ijk[D*i:D*(i+1),D*j:D*(j+1),D*k:D*(k+1)] = I[i,j,k]

		and M_ijk = 0 everywhere else.

		The 'unconstrained' class of keys is case sensitive.
		'''
		# Build a sparse matrix representation from the voxmap
		self._shape = None
		data, rows, cols = [], [], []
		M, N = 0, 0
		# Match unconstrained keys with optional scales
		unre = re.compile('^unconstrained(_([0-9]+)x)?')
		# Note that voxmap may not be a proper dictionary (e.g., NpzFile)
		for key in sorted(voxmap.keys()):
			# Check if the key is a scaled or unconstrained key
			m = unre.match(key)
			if m: scale = int(m.groups(1)[1])
			else: scale = 1

			v = np.asarray(voxmap[key]).astype(np.float64)
			if v.ndim != 3:
				raise ValueError('All voxmaps must have three dimensions')

			# Scale the shape of the grid if appropriate
			shape = v.shape
			if scale != 1: shape = tuple(scale * sv for sv in shape)

			if not self._shape:
				self._shape = shape
				M = np.product(self._shape)
			elif self._shape != shape:
				raise ValueError('All voxmap shapes must be compatible')

			if not m or scale == 1:
				# Process constrained or scale-1 unconstrained maps
				v = v.ravel('C')
				ri = np.nonzero(v)[0]
				data.extend(v[ri])
				rows.extend(ri)

				if not m:
					# Constrained voxels share a column
					cols.extend(N for sv in range(len(ri)))
					N += 1
				else:
					# Unconstrained voxels get their own columns
					cols.extend(range(N, N + len(ri)))
					N += len(ri)
				continue

			# List of neighbor offsets for scaled voxels
			nbrs = list(iproduct(*(range(scale)
					for sv in range(v.ndim))))

			# Process each scaled voxel as a separate column
			for i, j, k in np.transpose(np.nonzero(v)):
				# Explode each voxel in a scaled unconstrained image
				# np.ravel_multi_index requires the transposed form
				vxi = np.transpose([[scale * i + ii,
							scale * j + jj,
							scale * k + kk]
						for ii, jj, kk in nbrs])
				# Map the exploded voxels to C-raveled indices
				ri = np.ravel_multi_index(vxi, shape, order='C')
				# Build the next block of the matrix
				# All voxels in the block get the same weight
				rows.extend(ri)
				cols.extend(N for sv in range(len(ri)))
				data.extend(v[i, j, k] for sv in range(len(ri)))
				N += 1

		# Confirm that the background has the right form
		self._s = np.array(s, dtype=np.float64)
		if self._s.shape and self._s.shape != self._shape:
			raise ValueError('Background slowness s must be scalar or match voxmap shapes')

		# Convert the representation to CSR for efficiency
		self._voxmap = coo_matrix((data, (rows, cols)), shape=(M, N)).tocsr()


	@property
	def nnz(self):
		'''The number of parameters in the slowness model'''
		return self._voxmap.shape[1]


	def clip(self, x, smin, smax):
		'''
		Clip the perturbation x so that self.perturb(x) clips to the
		range [smin, smax].
		'''
		raise NotImplementedError('Clipping is ill-defined for %s' % (self.__class__.__name__))


	def flatten(self, s):
		'''
		Project the array s into a 1-D array in the space of piecewise
		constant slowness values. The slowness s will be raveled in
		C-order, so its shape must be compatible with self.shape.
		'''
		return self._voxmap.T @ np.ravel(s, 'C')


	def unflatten(self, s):
		'''
		Expand the array s, which must be a scalar or have shape
		(self.nnz,), into an array of shape self.shape.
		'''
		shape = self.shape
		s = np.asarray(s)

		if not s.shape:
			os = s
			s = np.empty((self.nnz,), dtype=os.dtype)
			s[:] = os

		return (self._voxmap @ s).reshape(self.shape, order='C')


	def tosparse(self):
		'''
		Return a CSR representation of the linear transformation
		represented by this Slowness, which maps one of an arbitrary
		number of slowness values to each voxel type.
		'''
		return self._voxmap.copy()
