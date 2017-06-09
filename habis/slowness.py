'''
Classes for modeling slowness on a 3-D voxel grid as a function of some number
of parameters.
'''

import numpy as np
from scipy.sparse import csr_matrix

from itertools import izip

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


	def perturb(self, x):
		'''
		Perturb the reference slowness by adding x, which must either
		be a scalar or have shape compatible with self.shape. The input
		x will be reshaped in row-major order to match the voxel grid.
		'''
		# Expand perturbation onto voxel grid, then add reference
		out = self.unflatten(x) + self._s
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
		return np.array(1., dtype=np.float64)


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
	each parameter represents a particular slowness for a predetermined set
	of voxels.
	'''
	def __init__(self, voxmap, s):
		'''
		Initialize a PiecewiseSlowness instance where the voxel (i,j,k)
		in a 3-D grid has a slowness s[voxmap[i,j,k]].
		'''
		voxmap = np.asarray(voxmap)
		if not np.issubdtype(voxmap.dtype, np.integer):
			raise TypeError('Argument voxmap must have an integral type')
		if not voxmap.ndim == 3:
			raise ValueError('Argument voxmap must be a 3-D array')

		# Copy the unflatted shape of the slowness
		self._shape = voxmap.shape

		# Find the unique type codes and column indices for voxel values
		typ, cols = np.unique(voxmap.ravel('C'), return_inverse=True)

		M = np.prod(voxmap.shape)
		N = len(typ)

		# Ensure nonnegative values
		if typ[0] < 0: raise ValueError('Values in voxmap must be nonnegative')

		self._s = np.array(s, dtype=np.float64)
		if not self._s.shape:
			# Special case: reference slowness is a scalar
			self._s = np.array([s] * N, dtype=np.float64)
		elif self._s.ndim != 1:
			raise ValueError('Slowness array must be 1-D')

		if len(self._s) != N:
			raise ValueError('Number of slowness values must be %d' % (N,))

		# In linear operator, rows are sequential
		rows = np.arange(M)
		# Entries of operator are all unity
		data = np.ones((M,), dtype=np.float64)
		# Build the sparse map
		self._voxmap = csr_matrix((data, (rows, cols)), shape=(M,N))


	@property
	def nnz(self):
		'''The number of parameters in the slowness model'''
		return self._voxmap.shape[1]


	def perturb(self, x):
		'''
		Perturb the reference slowness by adding x, then expanding the
		slowness values onto the grid according to the predefined voxel
		map.
		'''
		return self.unflatten(self._s + x)


	def clip(self, x, smin, smax):
		'''
		Clip the perturbation x so that self.perturb(x) clips to the
		range [smin, smax].
		'''
		out = np.array(x)
		if not out.shape:
			out = np.empty((self.nnz,), dtype=out.dtype)
			out[:] = x
		out += self._s
		np.clip(out, smin, smax, out)
		out -= self._s
		return out


	def flatten(self, s):
		'''
		Project the array s into a 1-D array in the space of piecewise
		constant slowness values. The slowness s will be raveled in
		C-order, so its shape must be compatible with self.shape.
		'''
		nnz = self.nnz
		return self._voxmap.T.dot(np.ravel(s, 'C'))


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

		return self._voxmap.dot(s).reshape(self.shape, order='C')


	def tosparse(self):
		'''
		Return a CSR representation of the linear transformation
		represented by this Slowness, which maps one of an arbitrary
		number of slowness values to each voxel type.
		'''
		return self._voxmap.copy()
