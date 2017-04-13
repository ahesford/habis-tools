'''
Classes for modeling slowness on a 3-D voxel grid as a function of some number
of parameters.
'''

import numpy as np
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

		If s is a scalar, shape must be provided as a 3-tuple of
		positive integers. If s is a Numpy array, shape must be None.
		'''
		# Make a copy of the array (may be a scalar)
		self._s = np.array(s, dtype=np.float64)

		if not self._s.shape:
			if len(shape) != 3:
				raise ValueError('Slowness shape must be a 3-tuple')
			self._shape = tuple(int(s) for s in shape)
			if any(s != sv or s < 1 for s, sv in izip(self._shape, shape)):
				raise ValueError('Slowness shape must contain positive integers')
		else:
			if shape: raise ValueError('Cannot specify shape when s is an array')
			if self._s.ndim != 3: 
				raise ValueError('Slowness array must be 3-D')
			self._shape = None

	@property
	def shape(self):
		'''The shape of the slowness model'''
		return self._shape or self._s.shape

	@property
	def nnz(self):
		'''The number of parameters in the slowness model'''
		return np.prod(self.shape)


	@staticmethod
	def _buildoutput(out, shape):
		'''
		If out is a Numpy array, verify that out.shape matches shape
		and return out. If out is None, create a float64-array of the
		given shape. The argument out should not be anything besides a
		Numpy array or None.
		'''
		if out is None: return np.empty(shape, dtype=np.float64)
		if out.shape != shape:
			raise ValueError('Shape of out must be %s' % (shape,))
		return out


	def perturb(self, x, out=None):
		'''
		Perturb the reference slowness by adding x, which must either
		be a scalar or have shape compatible with self.shape. The input
		x will be reshaped in row-major order to match the voxel grid.

		If out is provided, it must be an array of shape self.shape
		that will hold the perturbed slowness. The contents of out may
		be corrupted if this method fails to complete successfully.

		The return value will be the perturbed slowness, which will be
		identical to out if out is provided.
		'''
		# Expand perturbation onto voxel grid, then add reference
		out = self.unflatten(x, out)
		out += self._s
		return out


	def clip(self, x, smin, smax, out=None):
		'''
		Clip the perturbation x so that self.perturb(x) clips to the
		range [smin, smax].

		If out is provided, it must be an array of shape self.shape
		that will hold the clipped perturbation. The contents of out
		may be corrupted if this method fails to complete successfully.

		The return value will be the clipped perturbation, which will
		be identical to out if out is provided.
		'''
		# Create or verify output array
		out = self.perturb(x, out)
		# Clip the perturbed slowness and subtract the background
		np.clip(out, smin, smax, out)
		out -= self._s
		return out


	def flatten(self, s, out=None):
		'''
		Flatten the array s, with shape self.shape, into a 1-D array.
		'''
		# Create or verify output array
		out = self._buildoutput(out, (self.nnz,))
		# Just unravel in C order
		out[:] = s.ravel('C')
		return out


	def unflatten(self, s, out=None):
		'''
		Unflatten the array s, which must be a scalar or have a shape
		compatible with (self.nnz,), into an array of shape self.shape.
		'''
		shape = self.shape
		# Create or verify output array
		out = self._buildoutput(out, shape)
		# Make sure non-scalar s has the right shape
		s = np.asarray(s)
		if s.shape: s = s.reshape(shape, order='C')
		out[:,:,:] = s
		return out


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
		# Initialize the background slowness
		super(MaskedSlowness, self).__init__(s, shape)
		# Copy the mask
		self._mask = np.array(mask, dtype=bool)
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


	def clip(self, x, smin, smax, out=None):
		'''
		Clip the perturbation x so that self.perturb(x) clips to the
		range [smin, smax] for True-masked voxels.

		If out is provided, it must be an array of shape (self.nnz,)
		that will hold the clipped perturbation. The contents of out
		may be corrupted if this method fails to complete successfully.

		The return value will be the clipped perturbation, which will
		be identical to out if out is provided.
		'''
		# Create or verify output array
		out = self._buildoutput(out, (self.nnz,))
		# Add the reference to the perturbation and clip
		rs = self._s[self._mask]
		out[:] = x + rs
		np.clip(out, smin, smax, out)
		# Subtract the reference
		out -= rs
		return out


	def flatten(self, s, out=None):
		'''
		Flatten the array s, with shape self.shape, into a 1-D array of
		shape (self.nnz,) by pulling only the True-masked values of s.
		'''
		# Create or verify output
		out = self._buildoutput(out, (self.nnz,))
		# Pull the True-masked values of s into out
		out[:] = s[self._mask]
		return out


	def unflatten(self, s, out=None):
		'''
		Expand the array s, which must be a scalar or have a shape
		(self.nnz,), into a 3-D grid where self.mask is True. Voxels
		masked False will be 0.
		'''
		# Create or verify output array
		out = self._buildoutput(out, self.shape)
		# Zero the entire array
		out[:,:,:] = 0
		out[self._mask] = s
		return out


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
		self._voxmap = np.array(voxmap)
		if not np.issubdtype(self._voxmap.dtype, np.integer):
			raise TypeError('Argument voxmap must have an integral type')
		if not self._voxmap.ndim == 3:
			raise ValueError('Argument voxmap must be a 3-D array')

		minv = np.min(self._voxmap)
		if minv < 0: raise ValueError('Values in voxmap must be nonnegative')

		maxv = np.max(self._voxmap)
		self._s = np.array(s, dtype=np.float64)

		if not self._s.shape:
			# Special case: reference slowness is a scalar
			self._s = np.array([s] * (maxv + 1), dtype=np.float64)
			return
		elif self._s.ndim != 1:
			raise ValueError('Slowness array must be 1-D')

		if maxv >= len(self._s):
			raise ValueError('Values in voxmap must be less than %d' % (len(self._s),))

	@property
	def shape(self):
		'''The shape of the slowness model'''
		return self._voxmap.shape

	@property
	def nnz(self):
		'''The number of parameters in the slowness model'''
		return len(self._s)


	def perturb(self, x, out=None):
		'''
		Perturb the reference slowness by adding x, then expanding the
		slowness values onto the grid according to the predefined voxel
		map.

		If out is provided, it must be an array of shape self.shape
		that will hold the expanded slowness. The contents of out may
		be corrupted if this method fails to complete successfully.

		The return value will be the perturbed slowness, which will be
		identical to out if out is provided.
		'''
		return self.unflatten(self._s + x, out)


	def clip(self, x, smin, smax, out=None):
		'''
		Clip the perturbation x so that self.perturb(x) clips to the
		range [smin, smax].

		If out is provided, it must be an array of shape (self.nnz,)
		that will hold the clipped perturbation. The contents of out
		may be corrupted if this method fails to complete
		successfully.

		The return value will be the clipped perturbation, which will
		be identical to out if out is provided.
		'''
		out = self._buildoutput(out, (self.nnz,))
		np.clip(self._s + x, smin, smax, out)
		out -= self._s
		return out


	def flatten(self, s, out=None):
		'''
		Project the array s, with shape self.shape, into a 1-D array in
		the space of piecewise constant slowness values.

		If out is provided, it must be an array of shape (self.nnz,)
		that will hold the projected slowness. The contents of out may
		be corrupted if this method fails to complete successfully.

		The return value will be the projected slowness, which will be
		identical to out if it is provided.
		'''
		nnz = self.nnz
		out = self._buildoutput(out, (nnz,))
		for i in xrange(nnz):
			out[i] = np.sum(s[self._voxmap == i])
		return out


	def unflatten(self, s, out=None):
		'''
		Expand the array s, which must be a scalar or have shape
		(self.nnz,), into an array of shape self.shape.
		'''
		shape = self.shape
		out = self._buildoutput(out, shape)
		s = np.asarray(s)
		if s.shape: out[:,:,:] = s[self._voxmap]
		else: out[:,:,:] = s
		return out
