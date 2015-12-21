'''
Routines for manipulating HABIS signals.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, math
import collections
from numpy import fft, linalg as nla
from scipy import linalg as la
from scipy.signal import hilbert
from scipy.stats import linregress
from operator import itemgetter, add
from itertools import groupby

from pycwp import cutil, mio, signal


class Window(tuple):
	'''
	A subclass of tuple restricting the form to (start, length), where each
	is a nonnegative integer exposed as a property with the corresponding
	name, along with a computed propety end = start + length.
	'''
	def __new__(cls, *args, **kwargs):
		'''
		Create a new instance of Window.
		'''
		if len(args) + len(kwargs) != 2:
			raise ValueError('Exactly two arguments must be specified')

		start = kwargs.pop('start', None)
		length = kwargs.pop('length', None)
		end = kwargs.pop('end', None)

		if len(kwargs) > 0:
			raise ValueError('Unrecognized keyword arguments')

		if len(args) > 0:
			if start is not None:
				raise ValueError('Window start cannot be positional and keyword argument')
			start = args[0]
		if len(args) > 1:
			# Must be None because kwargs would have been empty
			length = args[1]

		# Fill in missing values
		if start is None:
			start = end - length
		elif length is None:
			length = end - start

		from .formats import _strict_nonnegative_int
		start = _strict_nonnegative_int(start)
		length = _strict_nonnegative_int(length)
		return tuple.__new__(cls, (start, length))

	@property
	def start(self): return self[0]
	@property
	def length(self): return self[1]
	@property
	def end(self): return self[0] + self[1]


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
	def fromfile(cls, f):
		'''
		Attempt to read a single wavefrom from a file containing either
		a habis.formats.WaveformSet object or a binary array readable
		by pycwp.mio.readbmat.

		If the file is a WaveformSet, it must specify exactly one
		transmit index and one receive index. (The values of these
		indices are ignored.) The resulting waveform is the output of
		WaveformSet.getwaveform(rid, tid) for the only valid indices.

		If the file is a binary array, it is read by mio.readbmat()
		with the argument dim=1 to force 1-D interpretation. The
		automatic dtype detection rules apply.

		As per each of these dependent I/O functions, f can either be a
		file name or a file-like object.
		'''
		try:
			from .formats import WaveformSet
			# First assume the file is a WaveformSet
			wset = WaveformSet.fromfile(f)
			if wset.ntx != 1 or wset.nrx != 1:
				raise TypeError('WaveformSet contains more than one waveform')
			rid = wset.rxidx[0]
			tid = wset.txidx[0]
			return wset.getwaveform(rid, tid)
		except ValueError: pass

		# As a fallback, try a 1-D binary array
		return cls(signal=mio.readbmat(f, dim=1))


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
		from .formats import _strict_nonnegative_int
		value = _strict_nonnegative_int(value)
		if value < self.datawin.end:
			raise ValueError('Waveform length must meet or exceed length of data window')
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
		try: return Window(self._datastart, len(self._data))
		except (TypeError, AttributeError): return Window(0, 0)


	@datawin.setter
	def datawin(self, value):
		value = Window(*value)
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


	def store(self, f, waveset=True):
		'''
		Store the contents of the waveform to f, which may be a file
		name or a file-like object.

		If waveset is True, the waveform is first wrapped in a
		habis.formats.WaveformSet object using the fromwaveform()
		factory and then written using WaveformSet.store(f).  No
		additional arguments to store() are supported.

		If waveset is False, the full waveform is stored in a 1-D
		binary matrix by calling pycwp.mio.writebmat(s, f), where the
		signal s is the output of self.getsignal().
		'''
		if waveset:
			from .formats import WaveformSet
			WaveformSet.fromwaveform(self).store(f)
		else:
			pycwp.mio.writebmat(self.getsignal(forcecopy=False), f)


	def copy(self):
		'''
		Return a copy of this waveform.
		'''
		datawin = self.datawin
		data = self.getsignal(datawin, forcecopy=True)
		return Waveform(self.nsamp, data, datawin.start)


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

		if isinstance(key, int):
			# Wrap negative values
			if key < 0: key += nsamp
			# Check if the wrapped value is in range
			if key < 0 or key >= nsamp:
				raise ValueError('Sample indices must be in range [0, self.nsamp) or [-self.nsamp, 0)')

			# Shift to the data window
			key -= dstart
			if 0 <= key < dlength:
				# Return data sample inside the window
				return self._data[key]
			else:
				# Return zero outside of the data window
				return self.dtype.type(0.)
		elif isinstance(key, slice):
			# Generate indices in the data window
			idxgen = (i - dstart for i in range(*key.indices(nsamp)))
			return np.array([
				self._data[idx] if (0 <= idx < dlength) else 0
				for idx in idxgen], dtype=self.dtype)

		raise IndexError('Only integers and slices are valid indices')


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
		cwin = Window(cstart, end=cend)

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
	def __idiv(self, other): return self.__iscale(other, 'div')
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

		from .formats import _strict_nonnegative_int
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
		window = Window(*window)

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


	def extremum(self, mx=True):
		'''
		Return a pair (v, i) such that v = self[i] is the maximum (when
		mx is True) or the minimum (when mx is False) value in the
		Waveform. If there are multiple occurrences of the extremum,
		the first occurrence is returned.
		'''
		try:
			# Find the maximum index and value
			i = self._data.argmax() if mx else self._data.argmin()
			v = self._data[i]
		except TypeError:
			# If there is no data, the maximum is just 0
			return (0, 0)

		# Offset the index by the start of the data window
		return v, i + self._datastart


	def window(self, window=None, tails=None):
		'''
		Return a windowed copy of the waveform where, outside the
		window (start, length), the signal is zero. If tails is
		provided, it should be a sequence of length 2N, where the first
		N values will multiply the signal in the range [start:start+N]
		and the last N values mull multiply the signal in the range
		[start+length-N:start+length].
		'''
		if window is None: window = (0, self.nsamp)
		window = Window(*window)

		if tails is not None and len(tails) > window.length:
			raise ValueError('Length of tails should not exceed length of window')

		dwin = self.datawin

		try:
			# Find overlap between the data and output windows
			ostart, istart, wlen = cutil.overlap(dwin, window)
			oend = ostart + wlen
		except TypeError:
			# There is no overlap, return an empty signal
			return Waveform(self.nsamp)

		# Otherwise, copy the signal and zero regions outside window
		data = self._data.copy()
		data[:ostart] = 0.
		data[oend:] = 0.

		# If there are tails, apply them
		if tails is not None:
			def tailer(data, dwin, tail, twin):
				'''Apply the tail to the data'''
				try:
					# Find the overlap and apply the tail
					ostart, istart, wlen = cutil.overlap(dwin, twin)
					oend, iend = ostart + wlen, istart + wlen
					data[ostart:oend] *= tail[istart:iend]
				except TypeError: return

			ltail = len(tails) / 2
			# Apply the left and right tails in succession
			lwin = (window[0], ltail)
			tailer(data, dwin, tails[:ltail], lwin)
			rwin = (window[0] + window[1] - ltail, ltail)
			tailer(data, dwin, tails[ltail:], rwin)

		# Return a copy of the signal, cropped to the window
		return Waveform(self.nsamp, data[ostart:oend], dwin.start + ostart)


	def envelope(self):
		'''
		Return the envelope (the magnitude of the Hilbert transform) of
		the waveform as a new Waveform object.
		'''
		if not self.isReal:
			raise TypeError('Envelope only works for real-valued signals')

		# Find the envelope in the data window
		env = np.abs(hilbert(self._data))
		# Encapsulate in a new Waveform object
		return Waveform(self.nsamp, env, self._datastart)


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
			n = self.nsamp if not real else (int(self.nsamp // 2) + 1)

		# Build the spectral indices
		kidx = np.arange(n)
		# Correct negative indices in a C2C transform
		if not real: kidx[int((n + 1) / 2):] -= n

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
		window = Window(*window)

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
		l2 = cutil.ceilpow2(l)

		# Compute the FFT of the (padded) data window
		ssig = self.fft((s, l2)).getsignal()

		if self.isReal:
			# R2C IDFT handles padding automatically
			csig = fft.irfft(ssig, n * l2)[:n*l]
		else:
			nex = n * l2
			isig = np.zeros((nex,), dtype=ssig.dtype)
			kmax, kmin = int((l2 + 1) / 2), -int(l2 / 2)
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
		shargs = {key: kwargs[key]
				for key in ['dtype'] if key in kwargs}
		deargs = {key: kwargs[key]
				for key in ['osamp', 'negcorr', 'wrapneg'] if key in kwargs}

		if deargs.get('negcorr', False):
			raise ValueError('Convenience alignment not supported for negcorr=True')

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


	def shift(self, d, dtype=None):
		'''
		Return a copy of the waveform cyclically shifted by a number of
		(optionally fractional) samples d using a spectral multiplier.
		Positive shifts correspond to delays (i.e., the signal moves
		right). The shifted signal is converted to the specified dtype,
		if provided; otherwise the shifted signal is of the same type
		as the original.

		Fourier transforms are always computed over the entire signal.
		'''
		# Try using the faster integer shift if possible
		try: return self.__intshift(d, dtype)
		except TypeError: pass

		# Determine whether the FFT will be R2C or C2C
		r2c = self.isReal
		ifftfunc = fft.irfft if r2c else fft.ifft

		n = self.nsamp

		fsig = self.fft().getsignal(forcecopy=False)
		nsig = len(fsig)

		# Build the spectral indices
		kidx = self.specidx(nsig, r2c)

		# Build the shift operator
		sh = np.exp(-2j * math.pi * d * kidx / n)
		# Correct the Nyquist frequency term for conjugate symmetry
		if n % 2 == 0: sh[n / 2] = np.real(sh[n / 2])

		if dtype is None: dtype = self.dtype
		ssig = ifftfunc(fsig * sh, n).astype(dtype)

		# Return a copy of the shifted signal
		return Waveform(self.nsamp, ssig, 0)


	def __intshift(self, d, dtype=None):
		'''
		Perform an integer shift without spectral manipulations by
		rewrapping data windows.
		'''
		if int(d) != d: raise TypeError('Shift amount must be an integer')
		dwin = self.datawin
		nsamp = self.nsamp

		# Prepare a new waveform
		shwave = Waveform(nsamp)

		# Any shift of a zero waveform is just a zero waveform
		if dwin.length == 0: return shwave

		if dtype is None: dtype = self.dtype

		# Wrap the shifted start into the waveform window
		nstart = int(dwin.start + d) % nsamp
		nend = nstart + dwin.length

		if nend <= nsamp:
			# If the shifted data window fits, no need to rewrap
			# Use astype to force a copy of the data
			data = self._data.astype(dtype)
			shwave.setsignal(data, nstart)
		else:
			# Shifted window will wrap; manually unwrap
			data = np.zeros((nsamp,), dtype=dtype)
			idxmap = [i % nsamp for i in range(nstart, nend)]
			data[idxmap] = self._data
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
		if end is None: end = int(self.nsamp / 2) + 1
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
		Perform cyclic cross-correlation of self and ref using DFTs. If
		osamp is provided, it must be a nonnegative power of two that
		specifies an oversampling rate for the signal. Sample index I
		at the input sampling rate corresponds to index I * osamp at
		the output sampling rate.

		All transforms are always rounded to a power of two. The
		cross-correlation signal will contain only the nonzero portion
		of the cross-correlation (self.nsamp + ref.nsamp - 1 samples),
		interpolated according to the oversampling rate.

		If ref is complex and self is not, the output dtype will be
		that of ref. Otherwise, the output dtype will be that of self.
		'''
		if osamp < 1:
			raise ValueError('Oversampling factor must be at least unity')
		if cutil.ceilpow2(osamp) != osamp:
			raise ValueError('Oversampling factor must be a power of 2')

		# Find the data windows that will be convolved
		sstart, slength = self.datawin
		try:
			rstart, rlength = ref.datawin
		except AttributeError:
			# Treat a Numpy array as a signal starting at sample 0
			ref = type(self)(signal=ref)
			rstart, rlength = ref.datawin

		# Find the necessary padding for convolution
		npad = cutil.ceilpow2(slength + rlength - 1)
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
			kmax, kmin = int((npad + 1) / 2), -int(npad / 2)
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
		return xcwave if ssamp == 0 else xcwave.shift(ssamp)


	def delay(self, ref, osamp=1, negcorr=False, wrapneg=False):
		'''
		Find the sample offset that maximizes self.xcorr(ref, osamp).
		This is interpreted as the delay, in samples, necessary to
		cyclically shift ref to align the signal with self.

		If negcorr is True, return a tuple (d, s), where d is the
		sample offset that maximizes the absolute value of the cross-
		correlation, and s is the sign (-1 or 1, integer) of the
		cross-correlation at this point.

		If wrapneg is True, negative delays are cyclically wrapped to
		positive delays within the reference window.
		'''
		# Ensure that the reference has a sample length
		try: _ = ref.nsamp
		except AttributeError: ref = type(self)(signal=ref)

		# Compute the cross-correlation
		xcorr = self.xcorr(ref, osamp).real

		# If the correlation is explicitly zero, the delay is 0
		if xcorr.datawin.length == 0: return 0

		# Find the point of maximal correlation
		if negcorr:
			t = abs(xcorr).extremum(mx=True)[1]
			# Determine the sign of the correlation
			s = int(math.copysign(1., xcorr[t]))
		else: t = xcorr.extremum(mx=True)[1]

		# Unwrap negative values and scale by the oversampling rate
		if t >= osamp * self.nsamp: t -= xcorr.nsamp
		# The convolution should be zero outside the minimally padded window
		if t <= -osamp * ref.nsamp:
			raise ValueError('Maximum cross-correlation found in region where values are expected to be zero')
		t /= float(osamp)
		# Wrap negative values back if desired
		if wrapneg: t %= ref.nsamp

		return ((t, s) if negcorr else t)


	def envpeaks(self, *args, **kwargs):
		'''
		Return the output of pycwp.signal.findpeaks for self.envelope().
		The args and kwargs are passed on to findpeaks().
		'''
		# Find peaks in the data window
		start, length = self.datawin
		envelope = self.envelope().getsignal((start, length), forcecopy=False)
		peaks = signal.findpeaks(envelope, *args, **kwargs)
		# Map indices to full sample window
		return [(v[0] + start, v[1], v[2], v[3]) for v in peaks]


	def isolatepeak(self, index=None, **kwargs):
		'''
		Use self.envpeaks() to identify the peak nearest the provided
		index that also matches any filtering criteria specified
		in the arguments. A copy of the signal, which is windowed to
		+/- 1 peak width around the identified peak, is returned.

		If index is None, the most prominent peak in the signal is
		returned.

		If kwargs contains "tails" or "window" arguments, they are
		stripped from the arguments and passed to self.window when
		windowing the peak. The window is relative to the position of
		the isolated peak; its starting index is added to the peak
		index to determine a true starting index.

		If kwargs contains no "window" argument, a default relative
		window of (-width, 2 * width) is assumed, where width is the
		width of the isolated peak.

		Any remaining **kwargs are passed to envpeaks.
		'''
		# Pull the tails and window arguments for the window function
		tails = kwargs.pop('tails', None)
		window = kwargs.pop('window', None)

		# Find the peak nearest the index
		peaks = self.envpeaks(**kwargs)
		if len(peaks) < 1: raise ValueError('No peaks found')

		if index is not None:
			ctr, _, width, _ = min(peaks, key=lambda pk: abs(pk[0] - index))
		else:
			ctr, _, width, _ = max(peaks, key=lambda pk: pk[3])

		# The default window is +/- 1 peak width
		if window is None: window = (-width, 2 * width)

		# Find the first and last samples of the absolute window
		fs = max(0, window[0] + ctr)
		ls = min(fs + window[1], self.nsamp)

		# Window the signal around the identified peak
		return self.window(Window(fs, end=ls), tails=tails)


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
			fmax =  int(n + 1) / 2
			fmin = -int(n / 2)

		# Apply the positive frequency window
		fsig[:start] = 0
		fsig[end:fmax] = 0

		if not r2c:
			# Apply the negative frequency window
			fsig[n+fmin:n-end+1] = 0
			fsig[n-start+1:] = 0

		# Apply the tails
		if len(tails) > 0:
			ltails = len(tails) / 2
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
	c = np.dot(np.conj(v[:cdeg,:].T), s * np.dot(np.conj(u[:,:cdeg].T), y))

	# Characterize the error and select the most significant contributors
	x = x0 + h * np.arange(len(y)).astype(complex)
	rms = [cutil.mse(y - cv * np.exp(x * lv), y) for lv, cv in zip(lm, c)]
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
