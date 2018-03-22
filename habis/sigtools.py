'''
Routines for manipulating HABIS signals.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys
import math
import numpy as np
import collections

import scipy

import functools

from numpy import linalg as nla
from scipy import linalg as la
from scipy.signal import hilbert, get_window
from scipy.stats import linregress
from operator import itemgetter, add
from itertools import groupby

from scipy.fftpack.helper import next_fast_len

try: import pyfftw
except ImportError: fft = np.fft
else: fft = pyfftw.interfaces.numpy_fft

from pycwp import cutil, signal, stats

from .formats import (WaveformSet, WaveformSetIOError,
			_strict_nonnegative_int, _strict_int)
from .stft import stft, istft

@functools.lru_cache(maxsize=32)
def _gabor_window(sigma, width, dtype=None):
	'''
	Using scipy.signal.get_window, build a Gaussian window with a standard
	deviation of sigma samples and a width of width samples.

	If dtype is not None, the resulting window will be cast into then
	appropriate dtype using numpy.ndarray.astype. If the dtype is not a
	subtype of np.inexact, the type of the window is not changed.
	Otherwise, the type of the window is converted to the type given by
	np.dtype(dtype).type(0).real.dtype.

	The results of this call are memoized by a 32-element cache.
	'''
	window = scipy.signal.get_window(('gaussian', sigma), width)
	if dtype is not None:
		# Make sure the type is a Numpy dtype
		dtype = np.dtype(dtype)
		# Convert the type if the input is an inexact
		if np.issubdtype(dtype, np.inexact):
			window = window.astype(dtype.type(0).real.dtype)
	return window


def _valley(lpk, hpk):
	'''
	For two peaks lpk and hpk as output by pycwp.signal.findpeaks, find a
	col between them associated with the smaller peak.

	If something goes wrong, None may be returned.
	'''
	# Pull the col from the smaller peak
	if lpk['peak'][1] > hpk['peak'][1]: lpk, hpk = hpk, lpk

	left, right = lpk['peak'][0], hpk['peak'][0]
	if left > right: left, right = right, left

	col = lpk['keycol']
	if col and left < col[0] < right: return col[0]

	col = lpk['subcol']
	if col and left < col[0] < right: return col[0]

	return None


class Window(tuple):
	'''
	A subclass of tuple restricting the form to (start, length), where each
	is a nonnegative integer exposed as a property with the corresponding
	name, along with a computed propety end = start + length.
	'''
	def __new__(cls, *args, **kwargs):
		'''
		Create a new Window instance.

		If exactly one positional argument is specified (and no keyword
		arguments except an optional 'nonneg'), it must either be a
		sequence of the form (start, length), or a dictionary-like map
		with exactly two keys 'start', 'length', or 'end'.

		If exactly two arguments (positional or keyword) are supplied
		apart from an optional 'nonneg' keyword-only argument, the
		first positional argument is the start and the second
		positional argument is the length. Missing positional arguments
		are populated from the keyword arguments.

		No more than two arguments apart from the optional 'nonneg'
		can be specified.

		If the keyword 'nonneg' is provided, it should be Boolean.
		When True, the start of the window is required to be positive.
		When False (the default), the start may be negative.
		'''
		# Remove nonneg from argument consideration
		nonneg = kwargs.pop('nonneg', False)

		if len(args) == 1 and len(kwargs) == 0:
			try:
				# Expand the positional tuple as kwargs
				kwargs = dict(args[0])
			except TypeError:
				try:
					# Expand the positional tuple
					args = tuple(args[0])
				except TypeError:
					raise TypeError('Single positional argument must be a sequence or dictionary')
			else:
				# For successful expansion to kwargs, clear args
				args = ()

		if len(args) + len(kwargs) != 2:
			raise TypeError("Exactly two arguments must be specified apart from optional 'nonneg'")

		start = kwargs.pop('start', None)
		length = kwargs.pop('length', None)
		end = kwargs.pop('end', None)

		if len(kwargs) > 0:
			raise TypeError('Unrecognized keyword %s' % (next(iter(kwargs.keys())),))

		if len(args) > 0:
			if start is not None:
				raise TypeError('Window start cannot be positional and keyword argument')
			start = args[0]
		if len(args) > 1:
			# Must be None because kwargs would have been empty
			length = args[1]

		# Fill in missing values
		try:
			if start is None: start = end - length
			elif length is None: length = end - start
		except TypeError:
			raise ValueError("Window start, length and end must be integers")

		try:
			if nonneg: start = _strict_nonnegative_int(start)
			else: start = _strict_int(start)
			length = _strict_nonnegative_int(length)
		except (ValueError, TypeError):
			raise ValueError('Window start and length must be integers '
						'and satisfy nonnegativity requirements')

		return tuple.__new__(cls, (start, length))

	@property
	def start(self): return self[0]
	@property
	def length(self): return self[1]
	@property
	def end(self): return self[0] + self[1]

	def shift(self, ds):
		'''
		Convenience function returning

			Window(self.start + ds, self.length).

		The start is allowed to be negative.
		'''
		return type(self)(self[0] + ds, self[1])


class Waveform(object):
	'''
	This class encapsulates a 1-D signal of an arbitrary data type. The
	waveform has an overall length and a "data window" contained within the
	overall length in which samples are explicitly stored. Outside of the
	"data window", the signal is assumed to be zero.
	'''
	@classmethod
	def empty_like(cls, wave):
		'''
		Return an empty waveform with the same "structure" as the
		provided waveform.
		'''
		return cls(wave.nsamp)


	@classmethod
	def fromfile(cls, f, *args, **kwargs):
		'''
		Attempt to read a Wavefrom from a file containing a single
		WaveformMap object. The WaveformMap must hold exactly one
		Waveform. (The transmit-receive key is ignored.)

		Extra arguments are passed to WaveformMap.load.
		'''
		wmap = WaveformMap.load(f, *args, **kwargs)
		if len(wmap) != 1:
			raise TypeError('WaveformMap must contain exactly one value')
		return next(iter(wmap.values()))


	def __init__(self, nsamp=0, signal=None, start=0):
		'''
		Initialize a signal of total length nsamp. If signal is
		provided, it is a 1-D (or 1-D compatible) Numpy array that
		specifies the nonzero data window of the waveform.

		If signal is provided, a data window is attached by calling
		self.setsignal(signal, start) after initialization.
		'''
		self._nsamp = nsamp
		self._datastart = 0
		self._data = None

		# Pull in the data signal if appropriate
		self.setsignal(signal, start)


	@property
	def nsamp(self):
		'''
		The "expected" length of the waveform.
		'''
		return self._nsamp

	@nsamp.setter
	def nsamp(self, value):
		value = _strict_nonnegative_int(value)

		# Make sure the change won't conflict with data window
		if value < self._nsamp and value < self.datawin.end:
			raise ValueError('Value of nsamp would truncate data')

		self._nsamp = value


	@property
	def dtype(self):
		'''
		The data type of the stored waveform. If no stored portion
		exists, the waveform is assumed to be float32.
		'''
		try: return self._data.dtype
		except AttributeError: return np.dtype('float32')


	@dtype.setter
	def dtype(self, value):
		try: self._data = self._data.astype(value)
		except AttributeError:
			raise AttributeError('Cannot set dtype on empty signal')


	@property
	def datawin(self):
		'''
		A 2-tuple of the form (start, length) that specifies the
		explicitly stored (nonzero) portion of the waveform.
		'''
		try: return Window(self._datastart, len(self._data), nonneg=True)
		except (TypeError, AttributeError): return Window(0, 0)


	@datawin.setter
	def datawin(self, value):
		value = Window(value, nonneg=True)
		if value.end > self.nsamp:
			raise ValueError('Specified window is not contained in Waveform')

		# Replace the data and window with the new segment
		sig = self.getsignal(value, forcecopy=False)
		self.setsignal(sig, value.start)


	@property
	def isReal(self):
		'''
		Returns False if the explicitly stored data window of the
		waveform is a complex type, True otherwise.
		'''
		return not np.issubdtype(self.dtype, np.complexfloating)


	@property
	def real(self):
		'''
		Return a new waveform whose data window is a view on the real
		part of this waveform's data window.
		'''
		try: return Waveform(self.nsamp, self._data.real, self._datastart)
		except AttributeError: return Waveform(self.nsamp)


	@property
	def imag(self):
		'''
		Return a new waveform whose data window is a view on the
		imaginary part of this waveform's data window.
		'''
		try: return Waveform(self.nsamp, self._data.imag, self._datastart)
		except AttributeError: return Waveform(self.nsamp)


	def store(self, f, compression=None, **kwargs):
		'''
		Store the contents of the waveform to f, which may be a file
		name or a file-like object. The waveform is first wrapped in a
		habis.formats.WaveformSet object using

		  WaveformSet.fromwaveform(self, copy=False, **kwargs)

		and written with WaveformSet.store(f, compression=compression).
		Because copy=False is set in WaveformSet.fromwaveform, this
		argument must not appear in kwargs.
		'''
		wset = WaveformSet.fromwaveform(self, copy=False, **kwargs)
		wset.store(f, compression=compression)


	def copy(self, copydata=True):
		'''
		Return a copy of this waveform. If copydata is True, the
		underlying data is copied; otherwise, only the surrounding
		structure is copied, while the data in the new Waveform is just
		a reference to the existing data.
		'''
		if copydata and self._data is not None:
			data = self._data.copy()
		else: data = self._data

		return Waveform(self._nsamp, data, self._datastart)


	def __iter__(self):
		'''
		Iterate over all samples in the waveform.

		*** DO NOT MUTATE THIS INSTANCE WHILE ITERATING ***
		'''
		# Create the default value for out-of-window samples
		zero = self.dtype.type(0.)

		# Figure the number of initial zeros, data samples, and final zeros
		ninitzero, ndata = self.datawin
		nfinzero = self.nsamp - ninitzero - ndata

		# Yield initial zeros, then data, then final zeros
		for i in range(ninitzero): yield zero

		# Now yield samples in the data window
		for i in range(ndata): yield self._data[i]

		# Now yield zeros until the end of the waveform
		for i in range(nfinzero): yield zero


	def __len__(self):
		'''
		Returns the number of samples in the waveform.
		'''
		return self.nsamp


	def __getitem__(self, key):
		'''
		For an integer index, return the waveform sample at that index.

		For a slice, return an ndarray containing the specified samples.
		'''
		dstart, dlength = self.datawin
		nsamp = self.nsamp

		errmsg = 'Only integers and slices are valid indices'

		try:
			# Make sure the key is numeric
			ikey = int(key)
		except TypeError:
			# Generate indices in the data window from a slice
			try: idxgen = (i - dstart for i in range(*key.indices(nsamp)))
			except AttributeError: raise IndexError(errmsg)

			# Copy the data in the slicer
			arr = [ ]
			for idx in idxgen:
				arr.append(self._data[idx] if 0 <= idx < dlength else 0)
			return np.array(arr, dtype=self.dtype)

		if ikey != key: raise IndexError(errmsg)

		# Wrap negative values
		if key < 0: key += nsamp
		# Check if the wrapped value is in range
		if key < 0 or key >= nsamp:
			raise ValueError('Sample indices must be in range [-self.nsamp, self.nsamp)')

		# Shift to the data window
		key -= dstart
		if 0 <= key < dlength:
			# Return data sample inside the window
			return self._data[key]
		else:
			# Return zero outside of the data window
			return self.dtype.type(0.)


	def __pos__(self):
		return self.copy()

	def __neg__(self):
		datawin = self.datawin
		data = -self.getsignal(datawin, forcecopy=False)
		return Waveform(self.nsamp, data, datawin.start)

	def __abs__(self):
		datawin = self.datawin
		data = abs(self.getsignal(datawin, forcecopy=False))
		return Waveform(self.nsamp, data, datawin.start)


	def __addsub(self, other, ssign=1, osign=1, inplace=False):
		'''
		Compute sgn(ssign) * self + sgn(osign) * other, where other is
		a waveform and waveform-like object.

		If inplace is True, the operation is done in-place on self.
		(The data window for self will be expanded if necessary.)
		Otherwise, a new Waveform will be created and returned.
		'''
		# Grab the data window for this signal
		dwin = self.datawin

		try:
			owin = other.datawin
		except AttributeError:
			# Try convering other to a waveform
			# This assumes that other starts at sample 0
			other = type(self)(signal=other)
			owin = other.datawin

		# Find the common data window
		if owin.length < 1:
			cstart, cend = dwin.start, dwin.end
		elif dwin.length < 1:
			cstart, cend = owin.start, owin.end
		else:
			cstart = min(dwin.start, owin.start)
			cend = max(dwin.end, owin.end)
		cwin = Window(cstart, end=cend, nonneg=True)

		# Grab other signal over common window (avoid copies if possible)
		osig = other.getsignal(cwin, forcecopy=False)

		if inplace:
			# Expand the data window for in-place arithmetic
			self.datawin = cwin

			if ssign >= 0:
				if osign >= 0: self._data += osig
				else: self._data -= osig
			else: self._data = (osig if osign >= 0 else -osig) - self._data

			return self

		# Grab this signal over common window
		ssig = self.getsignal(cwin, forcecopy=False)

		# Combine signals with consideration to sign
		if ssign >= 0:
			data = (ssig + osig) if osign >= 0 else (ssig - osig)
		else:
			data = (osig - ssig) if osign >= 0 else (-osig - ssig)

		nsamp = max(self.nsamp, other.nsamp)
		return type(self)(nsamp, data, cwin.start)


	def __add__(self, other): return self.__addsub(other, 1, 1, False)
	def __radd__(self, other): return self.__addsub(other, 1, 1, False)
	def __iadd__(self, other): return self.__addsub(other, 1, 1, True)
	def __sub__(self, other): return self.__addsub(other, 1, -1, False)
	def __rsub__(self, other): return self.__addsub(other, -1, 1, False)
	def __isub__(self, other): return self.__addsub(other, 1, -1, True)


	def __scale(self, other, mode):
		'''
		Apply a scale factor other to the waveform according to mode:

			mode == 'mul': Multiply waveform by other
			mode == 'div': Classsic divide waveform by other
			mode == 'floordiv': Floor divide waveform by other
			mode == 'truediv': True divide waveform by other
		'''
		# Ensure that other is a scalar
		try: complex(other)
		except TypeError: return NotImplemented

		try:
			modefunc = dict(mul=np.multiply, floordiv=np.floor_divide,
					div=np.divide, truediv=np.true_divide)[mode]
		except KeyError:
			raise ValueError('Invalid mode argument')

		# Pull the data window and scale
		datawin = self.datawin
		data = modefunc(self.getsignal(datawin, forcecopy=False), other)
		return Waveform(self.nsamp, data, datawin.start)


	def __mul__(self, other): return self.__scale(other, 'mul')
	def __rmul__(self, other): return self.__scale(other, 'mul')
	def __div__(self, other): return self.__scale(other, 'div')
	def __truediv__(self, other): return self.__scale(other, 'truediv')
	def __floordiv__(self, other): return self.__scale(other, 'floordiv')

	def __iscale(self, other, mode):
		'''
		In-place version of __scale.
		'''
		# Ensure that other is a scalar
		try: complex(other)
		except TypeError: return NotImplemented

		# If there is no data window, take no action
		if self._data is None or len(self._data) < 1: return self

		# Scale the data in-place
		if mode == 'mul': self._data *= other
		elif mode == 'div': self._data /= other
		elif mode == 'truediv': self._data /= other
		elif mode == 'floordiv': self._data //= other
		else:
			raise ValueError('Invalid mode argument')

		return self


	def __imul__(self, other): return self.__iscale(other, 'mul')
	def __idiv__(self, other): return self.__iscale(other, 'div')
	def __itruediv__(self, other): return self.__iscale(other, 'truediv')
	def __ifloordiv__(self, other): return self.__iscale(other, 'floordiv')


	def setsignal(self, signal, start=0):
		'''
		Replace the waveform's data window with the provided signal.
		The signal must either be a 1-D compatible array or "None" to
		clear the waveform data window. The value start, if provided,
		is the first sample.

		If signal is a Numpy ndarray, the retained data window may be a
		view to the provided signal rather than a copy.
		'''
		if signal is None or len(signal) < 1:
			self._data = None
			self._datastart = 0
			return

		# Ensure the start of the data window is always nonnegative
		start = _strict_nonnegative_int(start)

		# Ensure signal is 1-D
		self._data = dimcompat(signal, 1)
		self._datastart = start

		# Increase the length of nsamp if data window is too large
		length = len(self._data)
		if start + length > self.nsamp:
			self.nsamp = start + length


	def getsignal(self, window=None, dtype=None, forcecopy=True):
		'''
		Return, as a Numpy array, the contents of the waveform in the
		provided (start, length) window. A window of (0, self.nsamp) is
		assumed when None is specified.

		If dtype is provided, the output will be cast into the
		specified Numpy dtype. Otherwise, the Waveform dtype is used.

		If forcecopy is False, the returned array will be a view into
		the Waveform data array whenever the output window is wholly
		contained within self.datawin and dtype is None or the Waveform
		datatype. Otherwise, a new copy of the data will always be
		made. If the output window does not fall within the Waveform
		data window, or the datatypes are not the same, the output will
		always be a new copy.
		'''
		if window is None: window = (0, self.nsamp)
		window = Window(window)

		dwin = self.datawin

		# Find the datatype if necessary
		if dtype is None: dtype = self.dtype

		# Determine if the output is a view or a copy
		isview = ((not forcecopy) and dwin.start <= window.start
				and dwin.end >= window.end and dtype == self.dtype)

		try:
			# Find overlap between the data and output windows
			ostart, istart, wlen = cutil.overlap(window, dwin)
		except TypeError:
			# There is no overlap, the signal is 0
			return np.zeros((window.length,), dtype=dtype)

		oend, iend = ostart + wlen, istart + wlen

		if isview:
			return self._data[istart:iend]
		else:
			# Copy the overlapping portion
			signal = np.zeros((window.length,), dtype=dtype)
			signal[ostart:oend] = self._data[istart:iend]
			return signal


	def extremum(self, mx=True, window=None):
		'''
		Return a pair (v, i) such that v = self[i] is the maximum (when
		mx is True) or the minimum (when mx is False) value in the
		Waveform. If there are multiple occurrences of the extremum,
		the first occurrence is returned.

		If window is not None, it should be a sequence (start, length)
		or a Window object that indicates that starting index and
		length of the window over which the extremum will be sought.
		The length must be nonnegative, but the start can be any
		integer, with (start % self.nsamp) representing the actual
		starting sample. An interval that runs off the end of the
		signal wraps around to the beginning.

		If window is a tuple rather than a Window object, the start and
		length values will be truncated to integers.
		'''
		# Process limits
		wrapwin = None

		if window is None:
			window = (0, self._nsamp)
		else:
			start, length = window
			start = int(start) % self._nsamp
			if length < 0:
				raise ValueError('Window length must be nonnegative')

			if length >= self._nsamp:
				window = (0, self._nsamp)
			else:
				end = start + int(length)
				if end > self._nsamp:
					window = (start, self._nsamp - start)
					wrapwin = (0, end - self._nsamp)
				else: window = (start, end - start)

		dwin = (self._datastart, len(self._data))
		# With no data, there is no extremum; pick first sample
		if dwin[1] < 1: return (0, 0)

		expair = None
		sgn = 1 if mx else -1

		# Find overlap between interval and data in first window
		try:
			dst, wst, length = cutil.overlap(dwin, window)
		except TypeError:
			pass
		else:
			# Find extremum in relevant interval
			data = self._data[dst:dst+length]
			i = data.argmax() if mx else data.argmin()
			v = data[i]
			expair = v, i + dst

		# Find overlap between interval and data in wrapped window
		try:
			# This will raise TypeError if wrapwin is None
			dst, wst, length = cutil.overlap(dwin, wrapwin)
		except TypeError:
			pass
		else:
			data = self._data[dst:dst+length]
			i = data.argmax() if mx else data.argmin()
			v = data[i]
			# Update extremum of record if appropriate
			if not expair or sgn * v > sgn * expair[0]: expair = v, i + dst

		# If there is no extremum, the value and index are just 0
		if not expair: return (0, 0)

		# Offset the index by the start of the data window
		return expair[0], expair[1] + self._datastart


	def trim(self, start, end=None):
		'''
		A convience wrapper that returns the results of

		  self.window({'start': start, 'end': -end}, relative='datawin')

		provided that both start and end are provided. If end is None,
		end is replaced by start to produce a symmetric window.
		'''
		if end is None: end = start
		return self.window({'start': start, 'end': -end}, relative='datawin')


	def window(self, window, tails=0, relative=None):
		'''
		Return a windowed copy of the waveform where, outside the
		specified window (start, length), the signal is zero.

		The argument window can take three forms: a (start, length)
		tuple, a habis.sigtools.Window object, or a dictionary
		equivalent with exactly two of the keys 'start', 'end' or
		'length'.

		If tails is provided, it should be a scalar or a 1-D array of
		length 2N, where the first N values will multiply the signal in
		the range [start:start+N] and the last N values mull multiply
		the signal in the range [start+length-N:start+length]. If tails
		is a scalar, np.hanning(2 * tails) is used.

		Only when the window argument is a key-value map, the argument
		relative may take an optional value 'signal' or 'datawin'. When
		'signal' is specified, the value associated with any 'end' key
		is transformed according to

			end -> end + self.nsamp

		When 'datawin' is specified, any 'start' and 'end' keys are
		transformed according to

			start -> start + self.datawin.start
			end   -> end + self.datawin.end

		Any 'length' key of the map is always unchanged.
		'''
		try:
			if len(window) != 2:
				raise ValueError('Trigger enclosing exception')
		except (TypeError, ValueError):
				raise ValueError('The window must be a Window object, a two-element sequence, or a two-element key-value map')

		dwin = self.datawin

		try:
			# Test the dictionary-like nature of window, and make a copy
			window = dict(window)
		except TypeError:
			# If the window is not a dictionary, just make a Window object
			window = Window(window)
			if relative is not None:
				raise ValueError('Argument relative must be None if window is not dictionary-like')
		else:
			# Account for relative window options
			if relative == 'signal':
				try: window['end'] += self.nsamp
				except KeyError: pass
			elif relative == 'datawin':
				try: window['start'] += dwin.start
				except KeyError: pass
				try: window['end'] += dwin.end
				except KeyError: pass
			elif relative is not None:
				raise ValueError("Argument relative must be one of None, 'signal', or 'datawin'")
			# Create the window
			window = Window(window)

		tails = np.asarray(tails)
		if tails.ndim < 1:
			tails = np.hanning(2 * tails)
		elif tails.ndim > 1:
			raise TypeError('Tails must be scalar or 1-D array compatible')

		if len(tails) > window.length:
			raise ValueError('Length of tails should not exceed length of window')

		try:
			# Find overlap between the data and output windows
			ostart, istart, wlen = cutil.overlap(dwin, window)
		except TypeError:
			# There is no overlap, return an empty signal
			return Waveform(self.nsamp)

		# Otherwise, copy the signal in the new data window
		data = self._data[ostart:ostart+wlen].copy()
		ndwin = Window(ostart + dwin.start, wlen)

		# If there are tails, apply them
		if len(tails) > 0:
			ltail = len(tails) // 2
			# Apply the left and right tails in succession
			lwin = (window.start, ltail)
			self._applytails(data, ndwin, tails[:ltail], lwin)
			rwin = (window.end - ltail, ltail)
			self._applytails(data, ndwin, tails[ltail:], rwin)

		# Return a copy of the signal, cropped to the window
		return Waveform(self.nsamp, data, ndwin.start)


	@staticmethod
	def _applytails(data, dwin, tail, twin):
		'''
		For a data array that spans the window dwin = (dstart, dlen),
		and a tail stencil that spans the window twin = (tstart, tlen),
		scale (in place) the data by the tail in the overlapping
		region, if any.
		'''
		try: ostart, istart, wlen = cutil.overlap(dwin, twin)
		except TypeError: return
		data[ostart:ostart+wlen] *= tail[istart:istart+wlen]


	def envelope(self):
		'''
		Return the envelope (the magnitude of the Hilbert transform) of
		the waveform as a new Waveform object.
		'''
		if not self.isReal:
			raise TypeError('Envelope only works for real-valued signals')

		dstart, dlen = self.datawin
		if not dlen: return Waveform(self.nsamp)

		# Find the envelope in the data window
		env = np.abs(hilbert(self._data))
		# Encapsulate in a new Waveform object
		return Waveform(self.nsamp, env, dstart)


	def specidx(self, n=None, real=None):
		'''
		Return the spectral indices, in FFT order, that correspond to
		the samples of self.fft().

		The parameter n determines the length of the transform window.
		If n is None, self.nsamp is assumed.

		When the parameter real is True, only positive indices are
		returned; otherwise, both positive and negative indices are
		returned. If real is None, self.isReal is assumed.
		'''
		if real is None: real = self.isReal
		if n is None:
			n = self.nsamp if not real else (self.nsamp // 2 + 1)

		# Build the spectral indices
		kidx = np.arange(n)
		# Correct negative indices in a C2C transform
		if not real: kidx[int(n + 1) // 2:] -= n

		return kidx


	def ifft(self, n=None, real=False):
		'''
		Returns the inverse FFT of the waveform. If real is True, the
		function numpy.fft.irfft is used; otherwise, numpy.fft.ifft is
		used. The value n has the same meaning as in these functions.
		'''
		fftfunc = fft.irfft if real else fft.ifft
		sig = fftfunc(self.getsignal(forcecopy=False), n)
		return Waveform(signal=sig)


	def fft(self, window=None, real=None):
		'''
		Returns the FFT of a portion of the waveform defined by the
		2-tuple window = (start, length). If window is not provided,
		(0, self.nsamp) is assumed.

		If real is True, an R2C DFT is used. (Performing an R2C DFT on
		a complex-valued signal discards the imaginary part of the
		signal before transforming.) If real is False, a C2C DFT is
		used. If real is None, it is assumed False if the signal is of
		a complex type and True otherwise.
		'''
		if window is None: window = (0, self.nsamp)
		window = Window(window)

		if real is None: real = self.isReal

		# Choose the right function
		fftfunc = fft.rfft if real else fft.fft

		dwin = self.datawin

		# Short-circuit FFT if data and FFT windows don't overlap
		if (dwin.start >= window.end) or (dwin.end <= window.start):
			return Waveform(window.length)

		# Because FFT can pad the end, no need to read past data window
		acqlen = max(0, min(dwin.end - window.start, window.length))
		acqwin = Window(window.start, acqlen)
		# Grab the signal without copying of possible
		sig = self.getsignal(acqwin, forcecopy=False)

		return Waveform(signal=fftfunc(sig, n=window.length))


	def oversample(self, n):
		'''
		Oversample the waveform by an (integer) factor n.
		'''
		s, l = self.datawin
		l2 = next_fast_len(l)

		# Compute the FFT of the (padded) data window
		ssig = self.fft((s, l2)).getsignal()

		if self.isReal:
			# R2C IDFT handles padding automatically
			csig = fft.irfft(ssig, n * l2)[:n*l]
		else:
			nex = n * l2
			isig = np.zeros((nex,), dtype=ssig.dtype)
			kmax, kmin = int((l2 + 1) // 2), -int(l2 // 2)
			isig[:kmax] = ssig[:kmax]
			isig[kmin:] = ssig[kmin:]
			csig = fft.ifft(isig, nex)[:n*l]

		return Waveform(n * self.nsamp, n * csig, n * s)


	def aligned(self, ref, **kwargs):
		'''
		Return a copy of the signal aligned to the reference
		habis.sigtools.Waveform ref by calling self.shift() with the
		equivalent of ref.delay(self).

		The keyword arguments are scanned for extra arguments to
		shift() and delay(), which are passed as appropriate.
		'''
		# Build the optional arguments to shift() and delay()
		shargs = { 'dtype' }
		shargs = { key: kwargs.pop(key) for key in shargs.intersection(kwargs) }
		deargs = { 'osamp', 'wrapneg', 'window' }
		deargs = { key: kwargs.pop(key) for key in deargs.intersection(kwargs) }

		if kwargs:
			raise TypeError(f'Unsupported keyword argument "{next(iter(kwargs))}"')

		# Find the necessary delay
		try:
			delay = ref.delay(self, **deargs)
		except AttributeError:
			ref = type(self)(signal=ref)
			delay = ref.delay(self, **deargs)

		if self.nsamp < ref.nsamp:
			# Pad self for proper shifting when ref is longer
			dwin = self.datawin
			signal = self.getsignal(dwin, forcecopy=False)
			exwave = Waveform(ref.nsamp, signal=signal, start=dwin.start)
			return exwave.shift(delay, **shargs)
		else:
			return self.shift(delay, **shargs)


	def shift(self, d, dtype=None, cyclic=False):
		'''
		Return a copy of the waveform shifted to the right by d, a
		(possibly negative and possibly fractional) number of samples,
		using a spectral multiplier.

		The shifted signal is converted to the specified dtype,
		if provided; otherwise the shifted signal is of the same type
		as the original.

		Fourier transforms are always computed over the entire signal.

		If cyclic is False, portions of the signal that would shift
		outside of the window [0, self.nsamp) (i.e., portions that
		would wrap in a cyclic shift) are truncated. Otherwise, a
		cyclic shift is performed.
		'''
		# Try using the faster integer shift if possible
		try: return self._intshift(d, dtype, cyclic)
		except TypeError: pass

		# Determine whether the FFT will be R2C or C2C
		r2c = self.isReal
		ifftfunc = fft.irfft if r2c else fft.ifft

		n = self.nsamp

		if not -n < d < n:
			# Signal is entirely shifted out of range
			return Waveform(n)

		fsig = self.fft().getsignal(forcecopy=False)
		nsig = len(fsig)

		# Build the spectral indices
		kidx = self.specidx(nsig, r2c)

		# Build the shift operator
		sh = np.exp(-2j * math.pi * d * kidx / n)
		# Correct the Nyquist frequency term for conjugate symmetry
		if n % 2 == 0: sh[n // 2] = np.real(sh[n // 2])

		if dtype is None: dtype = self.dtype
		ssig = ifftfunc(fsig * sh, n).astype(dtype)

		# Truncate wrapped portions, if desired
		if not cyclic:
			d = int(d)
			if d >= 0:
				ssig = ssig[d:]
				start = d
			else:
				ssig = ssig[:d]
				start = 0
		else: start = 0

		# Return a copy of the shifted signal
		return Waveform(n, ssig, start)


	def _intshift(self, d, dtype=None, cyclic=False):
		'''
		Perform an integer shift without spectral manipulations by
		rewrapping data windows.

		The shift d must be an integer within the precision implied by
		an inexact dtype (which will take the value self.dtype when
		dtype is None). If the dtype is not inexact, double precision
		is assumed.

		If cyclic is False, portions of the shifted signal that would
		fall outside of the range [0, self.nsamp) will be truncated;
		otherwise, the signal is shifted cyclically.
		'''
		try: eps = np.finfo(dtype or self.dtype).eps
		except ValueError: eps = np.finfo('float64').eps

		if not abs(d - int(d)) < eps * abs(d):
			raise TypeError('Shift amount must be an approximate integer')

		d = int(d)
		dwin = self.datawin
		nsamp = self.nsamp

		# Prepare a new waveform
		shwave = Waveform(nsamp)

		# Any shift of a zero waveform is just a zero waveform
		if dwin.length == 0: return shwave

		if not cyclic:
			try: wst, dst, length = cutil.overlap((0, nsamp), dwin.shift(d))
			except TypeError: return shwave

			data = self._data[dst:dst+length]
			if dtype is not None: data = data.astype(dtype)

			shwave.setsignal(data, wst)
		else:
			# Wrap shifted waveform into window
			nstart = (dwin.start + d) % nsamp
			nend = nstart + dwin.length

			if dtype is None: dtype = self._data.dtype

			if nend <= nsamp:
				# Window will not wrap
				data = self._data.astype(dtype)
				shwave.setsignal(data, nstart)
			else:
				# Manually wrap the window
				data = np.zeros((nsamp,), dtype=dtype)
				# Start of old data moves to right part of signal
				nright = nsamp - nstart
				data[-nright:] = self._data[:nright]
				# End of old data moves to left part of signal
				nleft = nend - nsamp
				data[:nleft] = self._data[-nleft:]
				shwave.setsignal(data, 0)

		return shwave


	def zerotime(self, start=0, end=None):
		'''
		Attempt to find the time origin of this signal by determining,
		using linear regression, the slope of the phase angle of the
		signal. Shifting by the negative of this slope should eliminate
		linear phase variations in the phase angle.

		The regression will be performed over the range (start, end) of
		*POSITIVE* DFT frequency bin indices. If end is None, all
		positive frequencies are used.
		'''
		if end is None: end = int(self.nsamp // 2) + 1
		# Compute positive DFT frequencies
		freqs = -2.0 * math.pi * np.arange(start, end) / self.nsamp
		# Compute the unwrapped phase angle in the band of interest
		sft = self.fft().getsignal((start, end - start), forcecopy=False)
		ang = np.unwrap(np.angle(sft))
		# Perform the regression (ignore all but slope)
		slope = linregress(freqs, ang)[0]
		return slope


	def xcorr(self, ref, osamp=1):
		'''
		Perform cross-correlation of self and ref using DFTs. If osamp
		is provided, it must be a nonnegative regular number that
		specifies an oversampling rate for the signal. Sample index I
		at the input sampling rate corresponds to index I * osamp at
		the output sampling rate.

		All transforms are always rounded to a regular number. The
		cross-correlation signal will contain only the nonzero portion
		of the cross-correlation (self.nsamp + ref.nsamp - 1 samples),
		interpolated according to the oversampling rate.

		If ref is complex and self is not, the output dtype will be
		that of ref. Otherwise, the output dtype will be that of self.
		'''
		if osamp < 1:
			raise ValueError('Oversampling factor must be at least unity')
		if next_fast_len(osamp) != osamp:
			raise ValueError('Oversampling factor must be a regular number')

		# Find the data windows that will be convolved
		sstart, slength = self.datawin
		try:
			rstart, rlength = ref.datawin
		except AttributeError:
			# Treat a Numpy array as a signal starting at sample 0
			ref = type(self)(signal=ref)
			rstart, rlength = ref.datawin

		# Find the necessary padding for convolution
		npad = next_fast_len(slength + rlength - 1)
		# Find the oversampled convolution length
		nint = osamp * npad

		# Create an empty waveform to store the oversampled correlation
		xcwave = Waveform(osamp * (self.nsamp + ref.nsamp - 1))

		# If one of the signals is 0, the cross-correlation is 0
		if slength == 0 or rlength == 0: return xcwave

		# Can use R2C DFT if both signals are real
		r2c = self.isReal and ref.isReal

		# Convolve the data windows spectrally
		sft = self.fft((sstart, npad), r2c).getsignal(forcecopy=False)
		rft = ref.fft((rstart, npad), r2c).getsignal(forcecopy=False)
		cfsig = (sft * np.conj(rft))

		if r2c:
			# Interpolation and IDFT is simple for real signals
			csig = fft.irfft(cfsig, nint)
		elif npad == nint:
			# Without interpolation, complex IDFT is equally simple
			csig = fft.ifft(cfsig, nint)
		else:
			# Complex interpolation is a bit messier
			# Padding must be done in middle
			cpsig = np.zeros((nint, ), dtype=cfsig.dtype)
			# Find the bounds of the positive and negative frequencies
			kmax, kmin = int((npad + 1) // 2), -int(npad // 2)
			# Copy frequencies to padded array and take IDFT
			cpsig[:kmax] = cfsig[:kmax]
			cpsig[kmin:] = cfsig[kmin:]
			csig = fft.ifft(cpsig, nint)

		# Samples correspond to positive shifts up to (oversampled)
		# length of self; remaining samples are wrapped negative shifts
		esamp = osamp * slength
		# Ignore zero portions of the cross-correlation
		ssamp = osamp * (-rlength + 1)

		# Unwrap the convolution values in natural order
		# Also scale values by oversampling rate
		csig = osamp * np.concatenate([csig[ssamp:], csig[:esamp]], axis=0)

		# Shift convolution according to (oversampled) window offsets
		ssamp += osamp * (sstart - rstart)
		xcwave.setsignal(csig, 0)
		# Use the Waveform object to shift the convolution into place
		return xcwave if ssamp == 0 else xcwave._intshift(ssamp, cyclic=True)


	def delay(self, ref, osamp=1, window=None, negcorr=False, wrapneg=False):
		'''
		Find the sample offset that maximizes self.xcorr(ref, osamp).
		This is interpreted as the delay, in samples, necessary to
		cyclically shift ref to align the signal with self.

		If window is None, it should be a window suitable for passing
		to self.extremum that will be used to limit the search region
		of the cross-correlation. The start and length of the window
		should be specified as if osamp were unity; they will be scaled
		by osamp by this routine.

		If negcorr is True, return a tuple (d, s), where d is the
		sample offset that maximizes the absolute value of the cross-
		correlation, and s is the sign (-1 or 1, integer) of the
		cross-correlation at this point.

		If wrapneg is True, negative delays are cyclically wrapped to
		positive delays within the reference window.
		'''
		# Ensure that the reference has a sample length
		try: ref.nsamp
		except AttributeError: ref = type(self)(signal=ref)

		# Compute the cross-correlation
		xcorr = self.xcorr(ref, osamp).real

		# If the correlation is explicitly zero, the delay is 0
		if xcorr.datawin.length == 0: return 0

		if window is not None:
			wst, wln = window
			window = (wst * osamp, wln * osamp)

		# Find the point of maximal correlation
		if negcorr:
			t = abs(xcorr).extremum(mx=True, window=window)[1]
			# Determine the sign of the correlation
			s = int(math.copysign(1., xcorr[t]))
		else: t = xcorr.extremum(mx=True, window=window)[1]

		# Unwrap negative values and scale by the oversampling rate
		if t >= osamp * self.nsamp: t -= xcorr.nsamp
		# The convolution should be zero outside the minimally padded window
		if t <= -osamp * ref.nsamp:
			raise ValueError('Maximum cross-correlation found in region where values are expected to be zero')
		t /= float(osamp)
		# Wrap negative values back if desired
		if wrapneg: t %= ref.nsamp

		return ((t, s) if negcorr else t)


	def envpeaks(self):
		'''
		Return the output of pycwp.signal.findpeaks for self.envelope().
		'''
		# Find peaks in the data window
		env = self.envelope()
		peaks = signal.findpeaks(env._data)
		st, ln = env.datawin
		# Map peak and col indices to global window
		return [ { k: (v and (v[0] + st, v[1]) or None)
			   for k, v in pk.items() } for pk in peaks ]


	def signsquare(self):
		'''
		Return a copy of the waveform with each sample squared and
		multiplied by its sign (or, equivalently, each sample
		multiplied by its absolute value).
		'''
		return Waveform(self._nsamp,
				self._data * np.abs(self._data), self._datastart)


	def imer(self, avgwin, *args, raw=False, return_all=False, **kwargs):
		'''
		For the modified energy ratios returned by the call

			NER, FER = self.mer(*args, **kwargs),

		compute the improved modified energy ratio, defined as

			IMER = MA(FER, avgwin) - MA(NER, avgwin),

		where MA is a centered moving average of width avgwin. The
		moving average is not well defined for the first (avgwin - 1)
		samples; these undefined values will be replaced by the nearest
		valid average.

		If avgwin is None, 0 or 1, no rolling average will be
		performed.

		The return value of self.mer must return two valid MERs or a
		ValueError will be raised.

		If the keyword-only arugment raw is True, the raw data array,
		defined over a window self.datawin, will be returned;
		otherwise, the ratio will be packaged as a new Waveform object.

		If the keyword-only argument return_all is True, the return
		value will be a dictionary that contains the key 'imer' with
		the IMER object (a raw array or a Waveform object) as its
		associated value, plus other key-value pairs that result from
		the call to self.mer with matching "raw" and "return_all"
		keyword arguments.
		'''
		avgwin = int(avgwin or 0)
		if not 0 <= avgwin < self.datawin.length:
			raise ValueError('Argument "avgwin" must be None, or an '
					'integer in range [0, self.datawin.length)')

		# Compute the energy ratios
		enrs = self.mer(*args, raw=True, return_all=True, **kwargs)

		try:
			ner = enrs['nmer']
			fer = enrs['fmer']
			if fer is None: raise ValueError('No valid far MER')
		except (KeyError, ValueError):
			raise ValueError('Call to self.mer(*args, **kwargs) '
						'must return two valid MER sequences')

		# Compute centered moving average of each signal
		if avgwin > 1:
			vld = len(ner) - avgwin
			ast = avgwin // 2
			aed = ast + vld
			ner[ast:aed] = stats.rolling_mean(ner, avgwin)[:vld]
			ner[:ast] = ner[ast]
			ner[aed:] = ner[aed - 1]
			fer[ast:aed] = stats.rolling_mean(fer, avgwin)[:vld]
			fer[:ast] = fer[ast]
			fer[aed:] = fer[aed - 1]

		# Wrap the IMER in a Waveform if necessary
		imer = fer - ner
		if not raw: imer = Waveform(self.nsamp, imer, self._datastart)

		if not return_all: return imer

		if not raw:
			# Wrap intermediate results in Waveform objects
			enrs = { k: Waveform(self.nsamp, v, self._datastart)
					for k, v in enrs.items() }

		enrs['imer'] = imer
		return enrs


	def mer(self, erpow=3, sigpow=3, envelope=False, *args,
			raw=False, return_all=False, **kwargs):
		'''
		For the energy ratio(s) returned by the call

			self.enratio(*args, **kwargs),

		compute the modified energy ratio, which is

			MER = ER**erpow * abs(self)**sigpow

		for an energy ratio ER. If the second energy ratio returned by
		self.enratio is None, the second MER will also be None.

		If envelope is True, abs(self) in the definition of MER is
		replaced by self.envelope().

		If the keyword-only argument raw is True, the raw data arrays
		(or None if the second MER is None), defined over self.datawin,
		will be returned; otherwise, non-None ratios will be packaged
		as new Waveform objects.

		If the keyword-only argument return_all is True, the return
		value will be a dictionary with keys 'ner', 'fer', 'nmer',
		'fmer', where 'ner' and 'fer' are the near and far energy
		windows returned by self.enratio (wrapped in Waveform objects
		if raw is False) and 'nmer' and 'fmer' are, respectively, the
		associated modified energy ratios.

		If return_all is False, the return value will be a simple tuple
		of values (nmer, fmer).
		'''
		# Compute the energy ratios and raise the powers
		ner, fer = self.enratio(*args, raw=True, **kwargs)
		if not envelope: sig = np.abs(self._data)**sigpow
		else: sig = np.abs(hilbert(self._data))**sigpow
		nmer = ner**erpow * sig
		if fer is not None: fmer = fer**erpow * sig
		else: fmer = None

		if not raw:
			# Wrap raw MER values in Waveform objects
			nmer = Waveform(self.nsamp, nmer, self._datastart)
			if fmer is not None:
				fmer = Waveform(self.nsamp, fmer, self._datastart)
			if return_all:
				# Wrap energy ratios in Waveform objects if necessary
				ner = Waveform(self.nsamp, ner, self._datastart)
				if fer is not None:
					fer = Waveform(self.nsamp, fer, self._datastart)

		if not return_all:
			return nmer, fmer

		return { 'nmer': nmer, 'fmer': fmer, 'ner': ner, 'fer': fer }


	def enratio(self, prewin, postwin=None, vpwin=None, *, wrap=False, raw=False):
		'''
		Compute energy ratios for each sample of the signal, which is
		given as the ratio between the average energy in a window of
		width postwin samples following (and including) the sample
		under test to the average energy in a window of width prewin
		samples preceding the sample under test.

		For samples approaching the ends of the signal, the preceding
		and following windows will run past the data; the missing
		samples will be replaced with the average energy in the entire
		signal if wrap_edges is False, or will cyclically wrap if wrap
		is True.

		If postwin is None, it will be the same as prewin.

		If vpwin is not None, a second energy ratio will be calculated
		and returned. In the second ratio, the following window remains
		the same, but the preceding window has width vpwin and will be
		shifted left by prewin samples. (In other words, the early
		preceding window in the second ratio will end where the
		preceding window from the first ratio begins.)

		If vpwin is None or 0, the secondary ratio will be None.

		If the keyword-only argument raw is True, the raw data arrays
		(or None if the secondary ratio is None), defined over the
		window self.datawin, will be returned; otherwise, non-None
		ratios will be packaged as new Waveform objects.
		'''
		prewin = int(prewin)
		if prewin <= 0: raise ValueError('Value "prewin" must be positive')

		postwin = int(postwin or prewin)
		if postwin <= 0:
			raise ValueError('Value "postwin" must be None or positive')

		vpwin = int(vpwin or 0)
		if vpwin < 0:
			raise ValueError('Value "vpwin" must be None or nonnegative')

		# Build an expanded data window
		ld = len(self._data)
		# The start and end of real data in expanded window
		start = prewin + vpwin
		end = ld + start
		exdata = np.empty((end + postwin - 1), dtype=self.dtype)
		# Copy the real data
		exdata[start:end] = self._data**2
		# Pad out-of bound values
		if wrap:
			# Handle left-edge wraps
			nrem = start
			while nrem > 0:
				edge = exdata[max(start,end-nrem):end]
				ledge = len(edge)
				exdata[nrem-ledge:nrem] = edge
				nrem -= ledge
			# Handle right-edge wraps
			nrem = end
			while nrem < len(exdata):
				ne = len(exdata) - nrem
				edge = exdata[start:min(start+ne,end)]
				ledge = len(edge)
				exdata[nrem:nrem+ledge] = edge
				nrem += ledge
		else:
			# Just use average energy levels outside of signals
			mval = np.mean(exdata[start:end])
			exdata[:start] = mval
			exdata[end:] = mval

		# Compute the rolling average for the far-left window
		if vpwin: vleft = stats.rolling_mean(exdata, vpwin)
		else: vleft = None

		# When left and far-left match, reuse the averages
		# Either way, skip far-left-only samples in data or averages
		if prewin == vpwin: left = vleft[vpwin:]
		else: left = stats.rolling_mean(exdata[vpwin:], prewin)

		# When left or far-left match right, reuse
		# Regardless, skip far-left- and left-only samples
		if prewin == postwin: right = left[prewin:]
		elif prewin == postwin: right = vleft[start:]
		else: right = stats.rolling_mean(exdata[start:], postwin)

		ner = right[:ld] / left[:ld]
		if vpwin: fer = right[:ld] / vleft[:ld]
		else: fer = None

		if not raw:
			# Wrap raw energy ratios in Waveform objects
			ner = Waveform(self.nsamp, ner, self._datastart)
			if fer is not None:
				fer = Waveform(self.nsamp, fer, self._datastart)

		return ner, fer


	def isolatepeak(self, index=None, **kwargs):
		'''
		Use self.envpeaks() to identify the peak nearest the provided
		index that also matches any filtering criteria specified in the
		kwargs. A copy of the signal, windowed (by default) to +/-1
		peak width around the identified peak, is returned, along with
		the location of the identified peak.

		If index is None, the most prominent peak in the signal is
		returned.

		The kwargs can contain several filtering options:

		* minprom: Only peaks with a prominence greater than the
		  specified value are considered (default: 0).

		* prommode: One of 'absolute' (default), 'relative',
		  'noisedb', or 'rolling_snr'; changes the interpretation of
		  minprom. For 'absolute', the minprom value is interpreted as
		  an absolute threshold. For 'relative', the minprom threshold
		  is specified as a fraction of the prominence of the most
		  prominent peak.  For 'noisedb', the minprom threshold
		  specifies a ratio, in dB, between the peak prominence and the
		  noise floor (computed using self.noisefloor); see the
		  'noisewin' kwarg. For 'rolling_snr', the minprom threshold
		  specifies the SNR as estimated by self.rolling_snr.

		* useheight: If False (default), the 'prominence' used to
		  filter peaks according to minprom and prommode is the
		  topographical prominence (the difference between the height
		  of the peak and the height of its key col); if True,
		  'prominence' is simply the height of the peak above 0.

		* noisewin: If prommode is 'noisedb', this optional kwarg
		  specifies the width of the window used to estimate noise
		  floor with self.noisefloor. If prommode is 'rolling_snr',
		  this argument is passed to self.rolling_snr to estimate the
		  SNR per sample. Defaults to 100 and is ignored when prommode
		  is not 'noisedb' or 'rolling_snr'.

		* minwidth: Only peaks with a width (the distance between the
		  index of the peak and the index of the closer of its key or
		  sub cols) no less than the specified value are considered.

		* relaxed: If True (False by default), relax the width and
		  prominence constraints to ensure the highest peak (or peaks,
		  if there are two equally dominant peaks) is always isolated,
		  subject to satisfaction of the maxshift parameter.

		* maxshift: If specified, limits the maximum number of samples
		  the peak to isolate is allowed to fall from the provided
		  index. If the distance exceeds maxshift, a ValueError is
		  raised. Ignored when index is None.

		A single highest peak, which has no key col, has a width that
		reaches to the further end of the signal's data window and a
		prominence that equals its envelope amplitude.

		Two additional kwargs control the isolation window:

		* window: The string 'tight' or a tuple (rstart, length). The
		  actual isolation window of the tuple form will be, for an
		  index pidx of the isolated peak, (rstart + pidx, length).
		  The default relative window is (-width, 2 * width).

		  If the window is 'tight', the window will span from the
		  lowest point between the peak and any other peak (not just a
		  higher one) to the left, to the lowest point between the peak
		  and any other peak to the right.

		  If, in 'tight' mode, there is no other peak to the left
		  (right) of the isolated peak, the window will not clip the
		  signal to the left (right).

		* tails: Passed through as the "tails" argument to
		  self.window() without further processing.
		'''
		# Pull the tails and window arguments for the window function
		tails = kwargs.pop('tails', 0)
		window = kwargs.pop('window', None)

		# Pull filtering options
		minprom = kwargs.pop('minprom', 0)
		prommode = kwargs.pop('prommode', 'absolute')
		noisewin = kwargs.pop('noisewin', 100)
		minwidth = kwargs.pop('minwidth', 0)
		maxshift = kwargs.pop('maxshift', None)
		useheight = kwargs.pop('useheight', False)
		relaxed = kwargs.pop('relaxed', False)

		if len(kwargs):
			raise TypeError("Unrecognized keyword '%s'" % (next(iter(kwargs.keys())),))

		if prommode not in ('absolute', 'noisedb', 'rolling_snr', 'relative'):
			raise ValueError("Argument 'prommode' must be 'rolling_snr', "
						"'noisedb', 'absolute' or 'relative'")

		# Find peaks in the envelope, keyed by smaple index
		peaks = { pk['peak'][0]: pk for pk in self.envpeaks() }

		# Adjust the minimum prominence for relative thresholds
		if prommode == 'noisedb':
			# Compute the noise standard deviation
			try:
				nfloor  = self.noisefloor(noisewin)
			except ValueError:
				raise ValueError(f'Noise window {noisewin} too wide to compute standard deviation')
			minprom = 10.**((minprom + nfloor) / 20.)
		elif prommode == 'relative':
			try: minprom *= max(pk['peak'][1] for pk in peaks.values())
			except ValueError: raise ValueError('No peaks found')
		elif prommode == 'rolling_snr':
			try:
				srms = stats.rolling_rms(self._data, noisewin)
				if not len(srms): raise ValueError('Data too short')
			except ValueError:
				raise ValueError(f'Noise window {noisewin} too wide to compute rolling RMS')
			minprom = 10.**(minprom / 20.)

		dst, dlen = self.datawin

		# Identify peaks that match the desired criteria
		fpeaks = []
		maxpks = []
		for pk in peaks.values():
			i, v = pk['peak']

			try:
				ki, kv = pk['keycol']
			except TypeError:
				# Width of dominant peak is to far end of data window
				width = max(i, dlen - i)
				# Prominence of dominant peak is always its height
				prom = v
			else:
				# True prominence is height above key col
				if useheight: prom = v
				else: prom = v - kv

				try:
					si, sv = pk['subcol']
				except TypeError:
					# Width is to key col (only one)
					width = abs(i - ki)
				else:
					# Width is to closer of key and sub col
					width = min(abs(i - ki), abs(i - si))

			if prommode == 'rolling_snr':
				ip = i - noisewin - dst
				bsv = srms[max(0, ip - 1)]
				exceed = (prom >= minprom * bsv)
			else: exceed = (prom >= minprom)

			if width >= minwidth and exceed:
				fpeaks.append((i, prom, width))
			elif relaxed:
				# Remove all peaks smaler than this
				if maxpks and maxpks[0][1] < prom:
					maxpks = [ ]
				# Add this peak if there are no larger
				if not maxpks or maxpks[0][1] == prom:
					maxpks.append((i, prom, width))

		# In relaxed mode, replace an empty peak list with maximum peaks
		if not fpeaks and maxpks: fpeaks = maxpks

		if len(fpeaks) < 1: raise ValueError('No peaks found')

		if index is not None:
			ctr, _, width = min(fpeaks, key=lambda pk: abs(pk[0] - index))
			if maxshift is not None and abs(ctr - index) > maxshift:
				raise ValueError(f'Identified peak {ctr} too far '
							f'from expected location {index}')
		else:
			ctr, _, width = max(fpeaks, key=lambda pk: pk[1])

		if window != 'tight':
			# The default window is +/- 1 peak width
			if window is None: window = (-width, 2 * width)

			# Find the first and last samples of the absolute window
			fs = max(0, window[0] + ctr)
			ls = min(window[0] + window[1] + ctr, self.nsamp)
		elif window == 'tight':
			# Find the tight window in absolute samples
			lp, rp = None, None
			for pk in peaks.keys():
				if pk < ctr and (lp is None or lp < pk): lp = pk
				elif pk > ctr and (rp is None or rp > pk): rp = pk
			if lp is not None:
				fs = _valley(peaks[lp], peaks[ctr])
				if fs is None: raise ValueError('No col to left of peak')
			else: fs = 0
			if rp is not None:
				ls = _valley(peaks[ctr], peaks[rp])
				if ls is None: raise ValueError('No col to right of peak')
			else: ls = 0

		# Window the signal around the identified peak
		return self.window(Window(fs, end=ls), tails=tails), ctr


	def debias(self):
		'''
		Debias the Waveform so that the data window has zero mean.
		'''
		if self._data is not None:
			self._data -= np.mean(self._data)


	def gabor(self, sigma=8, trunc=5):
		'''
		Return the Gabor transform, as the output of habis.stft.stft,
		over the window self.datawin. The parameter sigma (measured in
		samples) specifies the standard deviation of the Gaussian
		window used in the STFT, and the overall window width will be
		int(2 * trunc * sigma).
		'''
		win = _gabor_window(sigma, int(2 * trunc * sigma), self.dtype)
		return stft(self._data, win)


	def denoise(self, band, noisewin, pfa, sigma=8, trunc=5, fs=1, floordb=None):
		'''
		Denoise the waveform based on Constant False Alarm Rate (CFAR,
		[1]) rejection in the Gabor spectrogram produced with a
		Gaussian window with a standard deviation of sigma samples,
		truncated to a total width of 2 * trunc * sigma samples. The
		paramter pfa is the probability of false alarm.

		The mean noise amplitude is calculated over the windows

		  band = (flo, fhi),
		  noisewin = (tlo, thi),

		where flo and fhi are the lower and upper limits, respectively,
		of the frequency bands to consider in the Gabor spectrogram;
		tlo and thi are the lower and upper time limits, respectively,
		in samples relative to the start of the signal (without regard
		to the start of the data window). Time samples outside of the
		data window are excluded from the averaging, even when the
		specified window extends beyond the data window.

		The sampling frequency of the signal is specified in fs, which
		is used to convert the values flo and fhi in the noise window
		into DFT bin indices.

		The noise window must satsify 0 <= flo < fhi <= fs / 2 and
		0 <= tlo < thi < self.nsamp.

		Noise is assumed to follow a Rayleigh distribution, with the
		scale parameter sigma = sqrt(2 / pi) * mean. All time-frequency
		bins with amplitudes A such that A / sigma < T, where the
		threshold T = sqrt(2 * log(1 / pfa)), are suppressed before the
		Gabor transform is inverted to return a denoised signal.
		Frequencies approximately below flo and fhi (as converted to
		DFT bin indices in integer arithmetic) will be explicitly
		suppressed regardless of their value.

		If floordb is not None, the denoised signal is augmented by
		adding Gaussian white noise (using numpy.random.normal) with an
		RMS level that is floordb dB relative to the peak ampltidue
		envelope of the original signal.

		Reference:
		[1] Chen and Qian, "CFAR detection and extraction of unknown
		    signal in noise with time-frequency Gabor transform", Proc.
		    SPIE 2762, Wavelet Applications III, 1996, pp. 285--294.
		'''
		# Check arguments for sanity
		flo, fhi = band
		tlo, thi = noisewin

		if not 0 <= flo < fhi <= fs:
			raise ValueError('Frequency bounds in noisewin must '
						'satisfy 0 <= flo < fhi <= fs')
		if not 0 <= tlo < thi <= self.nsamp:
			raise ValueError('Time bounds in noisewin must '
						'satisfy 0 <= tlo < thi <= self.nsamp')

		if not 0 < pfa < 1:
			raise ValueError('Parameter pfa must be in range (0, 1)')

		# Precompute the window for the Gabor transform
		win = _gabor_window(sigma, int(2 * trunc * sigma), self.dtype)

		# Compute STFT window size to interpret frequency limits
		nfft = next_fast_len(len(win))

		# Compute the Gabor transform and check noise window for sanity
		gtsig = stft(self._data, win, nfft=nfft)

		# Convert frequency limits to bin indices
		flo = int(nfft * flo / fs)
		fhi = int(nfft * fhi / fs)

		# Identify the noise band and interval in the transform
		try:
			fwin = (flo, max(fhi - flo, 1))
			flo, _, flen = cutil.overlap((0, gtsig.shape[1]), fwin)
			fhi = flo + flen
		except TypeError:
			raise ValueError('Frequency bounds in noisewin do '
						'not describe a valid spectral region')

		try:
			twin = (max(0, tlo - self._datastart), max(thi - tlo, 1))
			tlo, _, tlen = cutil.overlap((0, gtsig.shape[0]), twin)
			thi = tlo + tlen
		except TypeError:
			raise ValueError('Time bounds in noisewin do '
						'not describe a valid time region')

		# Eliminate out-of-band data
		if flo > 0: gtsig[:,:flo] = 0
		if fhi < gtsig.shape[1]: gtsig[:,fhi:] = 0

		# Estimate noise mean and convert to Rayleigh scale
		gtabs = np.abs(gtsig)
		nm = np.mean(gtabs[tlo:thi,flo:fhi])
		sigma = np.sqrt(2 / math.pi) * nm

		# Apply the CFAR threshold and invert
		T = np.sqrt(-2 * np.log(pfa))
		gtsig[gtabs <= T * sigma] = 0

		gw = istft(gtsig, win, nfft=nfft)

		if floordb is not None:
			emax = self.envelope().extremum()[0]
			scale = 10**(floordb / 20) * emax
			gw += np.random.normal(scale=scale, size=gw.shape)
		return Waveform(self.nsamp, gw, self._datastart)


	def bandpass(self, start, end, tails=0, dtype=None):
		'''
		Perform a bandpass operation that zeros all frequencies below
		the frequency bin at index start and above (and including) the
		frequency bin at index end. Only positive frequencies may be
		specified.

		If tails is provided, it should be a scalar or a 1-D array of
		length N that will modify the frequencies of sig according to
		the formula

			fsig[start:start+N/2] *= tails[:N/2],
			fsig[end-N/2:end] *= tails[-N/2:],

		where fsig = rfft(sig). If tails is a scalar, an array of
		np.hanning(2 * tails) is used. If the signal is complex (and,
		therefore, a C2C DFT is used, negative frequencies are
		similarly modified.

		If dtype is provided, the output is converted to the specified
		type; otherwise, the output has the same type as the input.
		'''
		# Check the window for sanity
		if start >= end:
			raise ValueError('Starting index should be less than ending index')

		# Check the tail for sanity
		tails = np.asarray(tails)
		if tails.ndim < 1:
			tails = np.hanning(2 * tails)
		elif tails.ndim > 1:
			raise TypeError('Tails must be scalar or 1-D array compatible')

		if len(tails) > (end - start):
			raise ValueError('Single-side tail should not exceed half window width')

		r2c = self.isReal
		n = self.nsamp

		# Transform the signal and apply the frequency window
		fsig = self.fft().getsignal(forcecopy=False)

		if r2c:
			fmax = len(fsig) + 1
		else:
			fmax =  (n + 1) // 2
			fmin = -(n // 2)

		# Apply the positive frequency window
		fsig[:start] = 0
		fsig[end:fmax] = 0

		if not r2c:
			# Apply the negative frequency window
			fsig[n+fmin:n-end+1] = 0
			fsig[n-start+1:] = 0

		# Apply the tails
		if len(tails) > 0:
			ltails = len(tails) // 2
			fsig[start:start+ltails] *= tails[:ltails]
			fsig[end-ltails:end] *= tails[-ltails:]

			if not r2c:
				tails = np.conj(tails)
				fsig[n-start:n-start-ltails:-1] *= tails[:ltails]
				fsig[n-end+ltails:n-end:-1] *= tails[-ltails:]

		ifftfunc = fft.irfft if r2c else fft.ifft
		if dtype is None: dtype = self.dtype
		bsig = ifftfunc(fsig, n).astype(dtype)

		return Waveform(self.nsamp, bsig, 0)


	def directivity(self, widths, theta):
		'''
		Return, as a new waveform, the result of applying to self a
		spectral directivity factor

			C(w) = exp(-widths(w) * sin(theta)),

		where w is a DFT bin index. The parameter widths must have the
		same length as self.fft().
		'''
		widths = np.asarray(widths)
		corr = np.exp(-widths * np.sin(theta))
		sigft = Waveform(signal=self.fft() * corr)
		return sigft.ifft(real=self.isReal)


	def noisefloor(self, noisewin):
		'''
		Estimate the noise floor, in dB, of the signal by sliding a
		window of width noisewin over the signal and identifying the
		minimum standard deviation (minstd). The noise floor is usually
		given by 20 * log10(minstd); however, if minstd is less than
		the floating-point epsilon for self.dtype (or, if self.dtype is
		an integer, for a 64-bit double) as indicated by np.finfo, the
		value of minstd will be replaced by the floating-point epsilon
		to estimate the noise floor.
		'''
		if len(self._data) < 2:
			raise ValueError('Signal data window must have at least 2 samples')

		# Find machine precision
		try: eps = np.finfo(self.dtype).eps
		except ValueError: eps = np.finfo('float64').eps

		# Compute minimum standard deviation and convert to dB
		mstd = min(stats.rolling_std(self._data, noisewin))
		return 20. * np.log10(max(mstd, eps))


	def rolling_snr(self, noisewin):
		'''
		Estimate the SNR (in dB) of the signal at each sample by
		comparing the amplitude of the envelope at that point to the
		RMS value of the preceding noisewin samples.

		For the first noisewin samples of the signal data, the RMS
		value of the preceding window is not well defined. The RMS
		value for the first noisewin samples will be used for all such
		early samples.
		'''
		if not 2 <= noisewin < len(self._data):
			raise ValueError('Data window and noisewin must be at least 2 samples')
		# Compute the rolling RMS levels
		rms = np.zeros_like(self._data)
		rms[noisewin:] = stats.rolling_rms(self._data, noisewin)[:-1]
		rms[:noisewin] = rms[noisewin]

		env = self.envelope()
		env._data = 20 * np.log10(env._data / rms)
		return env


	def snr(self, noisewin, rolling=False):
		'''
		Estimate the SNR (in dB) of the signal by comparing the peak
		envelope amplitude to the minimum standard deviation of the
		signal over a sliding window of width noisewin as computed by
		self.noisefloor.

		If rolling is True, instead of using self.noisefloor to
		estimate the noise level, the noise level for each sample is
		assumed to be the RMS value of the preceding noisewin samples.
		Note that, for samples within the first noisewin samples, this
		preceding interval is not well-defined. For all samples in this
		early region, the noise level is assumed to be the RMS value of
		the first noisewin samples of the signal.
		'''
		if not rolling:
			# Find noise floor and envelope peak
			nf = self.noisefloor(noisewin)
			env = max(np.abs(hilbert(self._data)))
			# Convert the peak amplitude, in dB, and subtract noise dB
			return 20. * np.log10(env) - nf
		else: return np.max(self.rolling_snr(noisewin)._data)


class WaveformMap(collections.abc.MutableMapping):
	'''
	A class to represent a mapping from transmit receive pairs (t, r) to
	Waveform objects, where t and r are both nonnegative integers, along
	with an overall sample count that defines the maximum time window
	(starting from an implicit zero-time reference) represented by all
	enclosed Waveforms.
	'''
	def __init__(self, nsamp=0, *args):
		'''
		Initialize a WaveformMap with the given sample duration and an
		optional mapping/iterable E = args[0] (if provided).
		'''
		# Copy the sample size
		self._nsamp = nsamp

		# Create a new underlying map
		self._wavemap = { }

		# Process any constructor arguments
		self.update(*args)


	@property
	def nsamp(self):
		'''
		The length of the time window, in samples, spanned by all
		Waveforms in the mapping.
		'''
		return self._nsamp


	@nsamp.setter
	def nsamp(self, nsamp):
		'''
		Ensure that the value of nsamp is a nonnegative integer and
		agrees with any existing records already in the mapping. If so,
		update the value.
		'''
		nsamp = _strict_nonnegative_int(nsamp)

		# No need to update the value if it matches
		if self._nsamp == nsamp: return

		try:
			# Try to update durations for all existing waveforms
			for wave in self._wavemap.values(): wave.nsamp = nsamp
		except ValueError:
			# Failure! Reset the durations
			for wave in self._wavemap.values(): wave.nsamp = self._nsamp
			raise
		else:
			# Success! Update this record
			self._nsamp = nsamp


	def __len__(self):
		'''
		Return the number of Waveform objects in this mapping.
		'''
		return len(self._wavemap)


	def __iter__(self):
		'''
		Return an iterator over the keys of the WaveformMap.
		'''
		return iter(self._wavemap)


	@classmethod
	def _verify_key(cls, key):
		'''
		Ensure that a key for mapping set/get/delete methods is a
		two-sequence of nonnegative integers, and return them in a
		tuple.

		If key is not of the expected form, a TypeError will be raised.
		'''
		try:
			t, r = key
			t = _strict_nonnegative_int(t)
			r = _strict_nonnegative_int(r)
		except Exception:
			raise TypeError(f'{cls.__name__} keys must be '
					'2-tuple of nonnegative integers')
		return t, r


	def __delitem__(self, key):
		'''
		Attempt to remove the provided key from the WaveformMap.
		'''
		del self._wavemap[self._verify_key(key)]


	def __getitem__(self, key):
		'''
		Get the Waveform object associated with the given key. A
		shallow copy of the Waveform is made (i.e., the underlying data
		may be shared with the internal value).
		'''
		return self._wavemap[self._verify_key(key)].copy(False)


	def __setitem__(self, key, value):
		'''
		Associate the value, which must be a Waveform object, to the
		key. If the value has an nsamp parameter that is larger than
		the WaveformMap nsamp parameter, the map's nsamp will be
		increased. Otherwise, the Waveform object's nsamp parameter
		will be increased.

		A shallow copy of the value will be made (i.e., the underlying
		data associatd with the input value may be shared with the
		stored Waveform object).
		'''
		#  Verify key and value, then make a shallow copy of the value
		key = self._verify_key(key)
		if not isinstance(value, Waveform):
			raise TypeError('Value must be a Waveform instance')
		value = value.copy(False)

		# Adjust sample window
		if self.nsamp > value.nsamp:
			value.nsamp = self.nsamp
		elif self.nsamp < value.nsamp:
			self.nsamp = value.nsamp

		# Record the waveform
		self._wavemap[key] = value


	def update(self, *args, **kwds):
		'''
		Override collections.abc.MutableMapping.update to enforce empty
		keyword arguments. Keyword arguments require string keys, which
		are incompatible with the 2-tuple of integers expected here.

		The behavior of update with positional-only arguments is
		identical to that of collections.abc.MutableMapping.update.
		'''
		if kwds:
			raise TypeError(f'{type(self).__name__}.update '
					'does not support keyword arguments')
		return super().update(*args)

	@classmethod
	def _serializer_prefixes(cls):
		'''
		A tuple (hdr, data) of name prefixes for the header and data
		portions of serialized WaveformMap representations.
		'''
		return ('com.habicoinc.WaveformMap.header',
				'com.habicoinc.WaveformMap.data')


	@classmethod
	def _store_chunk(cls, zf, hdr, data):
		'''
		A helper used by WaveformMap.store to write chunks of data.
		'''
		from uuid import uuid4
		from json import dumps

		# Identify this store operation
		storeid = uuid4().hex

		# Identifying prefix
		hpre, dpre = cls._serializer_prefixes()
		zf.writestr(f'{hpre}.{storeid}', dumps(hdr))
		zf.writestr(f'{dpre}.{storeid}', data)


	def store(self, f, compression=None, append=False):
		'''
		Serialize the contents of the WaveformMap to a specially
		crafted ZIP file given by f, which should be a string or a
		file-like object. The file should not be a ZipFile object,
		because it will be opened anew.

		The value of compression can be None, 'deflate', 'lzma' or
		'bzip2'. When opening the zipfile, this will determine the
		compression method used.

		If append is True, the ZipFile will be opened in append ('a')
		mode. Otherwise, the file will be opened in write ('w') mode
		that will truncate any existing output.

		In append mode, multiple WaveformMap objects may be serialized
		independently. The ZIP archive contains some header information
		and a collection of Waveform data for each store operation,
		indexed by a random UUID that should avoid conflicts.

		A single call to this method may behave as if the WaveformMap
		were arbitrarily split into several disjoint submaps, each
		written with an independent call WaveformMap.store with
		append=True.
		'''
		import zipfile

		try:
			compression = (compression or '').strip().lower()
			compression = { '': zipfile.ZIP_STORED,
					'deflate': zipfile.ZIP_DEFLATED,
					'bzip2': zipfile.ZIP_BZIP2,
					'lzma': zipfile.ZIP_LZMA }[compression]
		except KeyError:
			raise ValueError('Unrecognized compression scheme')

		if isinstance(f, zipfile.ZipFile):
			raise TypeError('Output file cannot be a ZipFile instance')

		byteorder = { 'little': '<', 'big': '>' }.get(sys.byteorder, 'unknown')

		# Only write 1024 records at a time
		chunksize = 1024

		zipmode = 'a' if append else 'w'
		with zipfile.ZipFile(f, mode=zipmode, compression=compression) as zf:

			# Start the header and the accumulation of bytes
			header = { 'nsamp': self.nsamp, 'records': [ ] }
			data = b''

			# Process each waveform record in turn
			for idx, ((t, r), wave) in enumerate(self.items()):
				if data and not idx % chunksize:
					# Write existing chunk
					self._store_chunk(zf, header, data)
					# Start with a fresh accumulation
					data = b''
					header['records'] = []

				record = { 't': t, 'r': r }

				dwin = wave.datawin
				record['datastart'] = dwin.start
				record['datalength'] = dwin.length

				dtype = wave.dtype
				record['typechar'] = dtype.char
				rbo = dtype.byteorder
				if rbo == '=' and byteorder != 'unknown':
					rbo = byteorder
				record['byteorder'] = rbo

				record['offset'] = len(data)
				data += wave.getsignal(dwin, forcecopy=False).tobytes()

				header['records'].append(record)

			# Write any remaining chunk
			if data: self._store_chunk(zf, header, data)


	@classmethod
	def _load_chunk(cls, header, data, nsamp, force_dtype, recid):
		'''
		A helper function used by WaveformMap.load to generate (t, r)
		pairs and Waveform objects from a serialized WaveformMap header
		and associated data block.
		'''
		# Process records in data-offset order to check contiguous access
		try:
			records = sorted(header['records'], key=lambda x: x['offset'])
		except (KeyError, TypeError):
			raise IOError(f'Header {recid} has no records or is malformed')

		# Track the final offset of the last record
		last_offset = 0
		for record in records:
			try:
				# Read the record header fields
				t, r = record['t'], record['r']
				dstart = record['datastart']
				dlen = record['datalength']
				typecode = record['byteorder'] + record['typechar']
				offset = record['offset']
			except KeyError:
				raise IOError(f'Malformed record {recid}')

			recpair = f'pair {t,r} of record {recid}'

			if offset != last_offset:
				raise IOError(f'Non-contiguous data offset in {recpair}')

			try:
				dtype = np.dtype(typecode)
			except TypeError:
				raise IOError(f'Unknown type {typecode} in {recpair}')

			# Convert the block of bytes to an array
			nbytes = dtype.itemsize * dlen
			try:
				dblock = data[offset:offset+nbytes]
				dblock = np.frombuffer(dblock, dtype=dtype, count=dlen)
			except ValueError:
				raise IOError(f'Unable to read data in {recpair}')

			if force_dtype is not None:
				dblock = dblock.astype(force_dtype)

			# Update the final offset
			last_offset += nbytes

			# Yield the waveform and transmit-receive indices
			yield (t,r), Waveform(nsamp, dblock, dstart)


	@classmethod
	def load(cls, f, dtype=None):
		'''
		Deserialize a WaveformMap instance from a specially crafted ZIP
		file given by f, which should be a string or a file-like
		object. The contents of f should have been produced by a prior
		call to WaveformMap.store. The file should not be a ZipFile
		object, because it will be opened anew.

		The contents of f may contain several serialized WaveformMap
		objects. These will be concatenated into a single instance. The
		largest value of "nsamp" encountered in the file will be
		assigned to the instance. If multiple records are associated
		with a single key, the most recently encoded record will be
		used.

		If dtype is not None, all waveforms will be coerced to the
		specified type after loading.

		An IOError will be raised if the contents of the input cannot
		be used to reconstruct any records.
		'''
		import zipfile, json

		if isinstance(f, zipfile.ZipFile):
			raise TypeError('Input file cannot be a ZipFile instance')

		if dtype is not None: dtype = np.dtype(dtype)

		# Create an empty map
		wmap = cls()

		# Parse the file
		with zipfile.ZipFile(f, mode='r') as zf:
			# Get the name prefixes with an added dot
			hpfx, dpfx = (c + '.' for c in cls._serializer_prefixes())
			# Build a list of matching header-data chunks
			chunks = collections.OrderedDict()
			for name in zf.namelist():
				if name.startswith(hpfx):
					recid = name.replace(hpfx, '')
					rectype = 'header'
				elif name.startswith(dpfx):
					recid = name.replace(dpfx, '')
					rectype = 'data'
				else: continue

				# Record the file for this chunk
				if recid not in chunks:
					chunks[recid] = { }
				if rectype in chunks[recid]:
					raise IOError(f'Duplicate {rectype} '
							'entry for record {recid}')
				chunks[recid][rectype] = name

			# Process each chunk that contains a header and data
			for recid, names in chunks.items():
				try:
					hname = names['header']
					dname = names['data']
				except KeyError:
					continue

				header = json.loads(zf.read(hname))
				data = zf.read(dname)

				try: nsamp = header['nsamp']
				except (KeyError, TypeError):
					raise IOError(f'Unable to parse nsamp from header {recid}')

				# Update the nsamp value if appropriate
				if wmap.nsamp < nsamp: wmap.nsamp = nsamp
				else: nsamp = wmap.nsamp

				# Process each waveform and update the map
				wmap.update(cls._load_chunk(header, data,
								nsamp, dtype, recid))

		return wmap


def dimcompat(sig, ndim=1):
	'''
	Given an input signal sig, ensure that it is compatible with an ndarray
	of dimension ndim. (That is, the number of dimensions is either ndim or
	every extra dimension has unity length.) If the array is compatible,
	return an ndarray squeezed to the proper dimensionality.

	If ndim is None, any dimensionality is satisfactory; sig is still
	squeezed and the output is guaranteed to be an ndarray.

	If the signal is incompatible, a TypeError is raised.
	'''
	# This will force a non-array into an array
	sig = np.squeeze(sig)

	if ndim is not None:
		if sig.ndim == 0:
			# Force 0-D squeezes to match dimensionality
			sig = sig[[np.newaxis]*ndim]
		elif sig.ndim != ndim:
			# Ensure squeezed dimensionality matches spec
			raise TypeError('Signal is not of dimension %d' % ndim)
	return sig


def aprony(maxdeg, x0, h, y, rcond=-1):
	'''
	Use an approximate Prony's method to fit at most maxdeg
	complex-weighted, complex exponentials to a signal y sampled according
	to the scheme y[i] = y(x0 + i * h). The complex exponentials are found
	by approximately inverting the Hankel system using a truncated SVD,
	where singular values not more than rcond times the largest singular
	value are discarded. If there are M singular values retained, a total
	of M complex exponentials are recovered.

	The return values are lm, c, rms, where the approximate signal yp is

		yp = reduce(add, (cv * exp(x * lv) for lv, cv in zip(lm, c)))

	for a sample vector x = x0 + h * arange(len(y)). The RMS error between
	yp and y is returned in rms.
	'''
	# Build the system to recover polynomial coefficients
	A = la.hankel(y[:-maxdeg], y[-maxdeg-1:])
	a = -A[:,:-1]
	b = A[:,-1]

	# Approximately invert the polynomial equation using truncated SVD
	s, res, adeg, sigma = nla.lstsq(a, b, rcond=rcond)

	# Find the roots of the polynomial, which yield exponential factors
	# Add unity high-order coefficient and reverse order (high-order first)
	u = np.roots(np.flipud(np.hstack((s,1))))
	# The roots yield the relevant exponential factors
	lm = np.log(u.astype(complex)) / h

	# Solve for the approximate amplitude coefficients using truncated SVD
	A = (u[None,:]**np.arange(len(y))[:,None]) * u[np.newaxis,:]**(x0 / h)
	u, s, v = la.svd(A)
	# Truncate to the approximate degree noted above, unless more
	# very small singular values need squashing
	cdeg = min(int(np.sum(s > s[0] * np.finfo(s.dtype).eps)), adeg)
	s = 1. / s[:cdeg]
	c = np.conj(v[:cdeg,:].T) @ (s * (np.conj(u[:,:cdeg].T) @ y))

	# Characterize the error and select the most significant contributors
	x = x0 + h * np.arange(len(y)).astype(complex)
	rms = [cutil.mse((y - cv * np.exp(x * lv)).flat, y.flat) for lv, cv in zip(lm, c)]
	rms, c, lm = [np.array(f) for f in zip(*sorted(zip(rms, c, lm))[-adeg:])]
	return lm, c, rms


def normalize(sig, env):
	'''
	Multiply the spectral response of the signal sig by the provided
	envelope env. The signal is padded (or truncated) to match the
	length of env before its FFT is computed. If the signal is padded, the
	output signal will be truncated to the length of the input signal.

	Both sig and env should be 1-D arrays or column vectors.
	'''
	sig = dimcompat(sig, 1)
	nsig = len(sig)
	fsig = fft.fft(sig, n=len(env))
	fsig *= env
	return fft.ifft(fsig)[:nsig].real


def getenvelope(sig, ref, thresh=0.01):
	'''
	Compute a complex spectral envelope to approximately convert the
	spectral characteristics of the input signal to the desired spectral
	characteristics in ref. The signal is padded (or truncated) to match
	the length of ref. The output envelope retains the modified length.

	Wherever the spectral magnitude of sig falls below the specified
	threshold thresh (relative to the peak amplitude), the envelope will
	have zero amplitude.

	Both sig and env should be 1-D arrays or column vectors.
	'''
	sig = dimcompat(sig, 1)
	ref = dimcompat(ref, 1)
	fsig = fft.fft(sig, len(ref))
	fsigmax = np.max(fsig)
	# Eliminate zeros to avoid complaints
	fsiga = (np.abs(fsig) > thresh * fsigmax).choose(1., fsig)
	return (np.abs(fsig) > thresh).choose(0., ref / fsiga)
