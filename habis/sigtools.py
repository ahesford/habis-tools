'''
Routines for manipulating HABIS signals.
'''

import numpy as np, math
from numpy import fft, linalg as nla
from scipy import linalg as la
from scipy.signal import hilbert
from operator import itemgetter, add
from itertools import groupby

from pycwp import cutil


class Waveform(object):
	'''
	This class encapsulates a 1-D signal of an arbitrary data type. The
	waveform has an overall length and a "data window" contained within the
	overall length in which samples are explicitly stored. Outside of the
	"data window", the signal is assumed to be zero.
	'''
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
		if signal is not None:
			self.setsignal(signal, start)


	@property
	def nsamp(self):
		'''
		The "expected" length of the waveform.
		'''
		return self._nsamp

	@nsamp.setter
	def nsamp(self, value):
		start, length = self.datawin
		if value < start + length:
			raise ValueError('Waveform length must meet or exceed length of data window')
		if value < 0:
			raise ValueError('Waveform length must be nonnegative')
		self._nsamp = value


	@property
	def dtype(self):
		'''
		The data type of the stored waveform. If no stored portion
		exists, the waveform is assumed to be float32.
		'''
		try: return self._data.dtype
		except AttributeError: return np.dtype('float32')


	@property
	def datawin(self):
		'''
		A 2-tuple of the form (start, length) that specifies the
		explicitly stored (nonzero) portion of the waveform.
		'''
		try: return (self._datastart, len(self._data))
		except (TypeError, AttributeError): return (0, 0)


	@property
	def isReal(self):
		'''
		Returns False if the explicitly stored data window of the
		waveform is a complex type, True otherwise.
		'''
		return not np.issubdtype(self.dtype, np.complexfloating)


	def copy(self):
		'''
		Return a copy of this waveform.
		'''
		return Waveform(self._nsamp, self._data.copy(), self._datastart)


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
			if key < 0 or key >= nsamp:
				raise ValueError('Sample indices must be in range [0, self.nsamp)')

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


	def setsignal(self, signal, start=0):
		'''
		Replace the waveform's data window with the provided signal.
		The signal must either be a 1-D compatible array or "None" to
		clear the waveform data window. The value start, if provided,
		is the first sample.

		If signal is a Numpy ndarray, the retained data window may be a
		view to the provided signal rather than a copy.
		'''
		if signal is None:
			self._data = None
			self._datastart = 0
			return

		if start < 0:
			raise ValueError('First sample of data window must be nonnegative')

		# Ensure signal is 1-D
		self._data = dimcompat(signal, 1)
		self._datastart = start

		# Increase the length of nsamp if data window is too large
		length = len(self._data)
		if start + length > self.nsamp:
			self.nsamp = start + length


	def getsignal(self, window=None, dtype=None):
		'''
		Return a new 1-D Numpy array populated with the contents of the
		waveform. If window is provided, it is a tuple of the form
		(start, length) that specifies the portion of the waveform to
		return. If window is None, (0, self.nsamp) is assumed.

		If dtype is provided, it is a Numpy datatype that will override
		the type of output array.
		'''
		if window is None: window = (0, self.nsamp)

		# Find the datatype if necessary
		if dtype is None: dtype = self.dtype

		signal = np.zeros((window[1],), dtype=dtype)

		try:
			# Find overlap between the data and output windows
			ostart, istart, wlen = cutil.overlap(window, self.datawin)
		except TypeError:
			# There is no overlap, the signal is 0
			return signal

		# Copy the overlapping portion
		oend, iend = ostart + wlen, istart + wlen
		signal[ostart:oend] = self._data[istart:iend]

		return signal


	def window(self, window=None, tails=None):
		'''
		Return a windowed copy of the waveform where, outside the
		window (start, length), the signal is zero. If tails is
		provided, it should be a sequence of length 2N, where the first
		N values will multiply the signal in the range [start:start+N]
		and the last N values mull multiply the signal in the range
		[start+length-N:start+length].
		'''
		if tails is not None and len(tails) > window[1]:
			raise ValueError('Length of tails should not exceed length of window')

		if window is None: window = (0, self.nsamp)

		datawin = self.datawin
		try:
			# Find overlap between the data and output windows
			ostart, istart, wlen = cutil.overlap(datawin, window)
			oend = ostart + wlen
		except TypeError:
			# There is no overlap, return an empty signal
			return Waveform(self.nsamp)
			return

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
			tailer(data, datawin, tails[:ltail], lwin)
			rwin = (window[0] + window[1] - ltail, ltail)
			tailer(data, datawin, tails[ltail:], rwin)

		# Return a copy of the signal
		return Waveform(self.nsamp, data, datawin[0])


	def envelope(self, window=None):
		'''
		Return the envelope, as the magnitude of the Hilbert transform,
		of the waveform over the (start, length) signal window. If
		window is None, (0, self.nsamp) is assumed.
		'''
		if window is None: window = (0, self.nsamp)

		# The real part of the signal type is the envelope type
		rtype = self.dtype.type().real.dtype

		try:
			# Find overlap between the data and output windows
			ostart, istart, wlen = cutil.overlap(window, self.datawin)
		except TypeError:
			# There is no overlap, the signal window and transform are 0
			return np.zeros((window[1],), dtype=rtype)

		oend, iend = ostart + wlen, istart + wlen

		if ostart < 1:
			# If the signal window starts before sample 1,
			# just use built-in Hilbert padding to expand
			sig = self._data[istart:iend]
			return np.abs(hilbert(sig, N=window[1]))
		else:
			# Otherwise, the signal window starts with some zeros
			# A new array must be created to hold the padding
			sig = np.zeros((window[1],), dtype=rtype)
			# Populate the output window with the envelope
			sig[ostart:oend] = np.abs(hilbert(self._data[istart:iend]))
			return sig


	def fft(self, window=None, real=None, inverse=False):
		'''
		Returns the FFT of a portion of the waveform defined by the
		2-tuple window = (start, length). If window is not provided,
		(0, self.nsamp) is assumed.

		If real is True, an R2C DFT is used. (Performing an R2C DFT on a
		complex-valued signal discards the imaginary part of the signal
		before transforming.) If real is False, a C2C DFT is used. If
		real is None, it is assumed False if the signal is of a complex
		type and True otherwise.

		If inverse is True, an inverse transform is computed. In this
		case, when real is True, a C2R inverse DFT is used. For all
		other values, a C2C inverse DFT is used.
		'''
		if window is None: window = (0, self.nsamp)

		# If real is not specified, determine from the waveform data type
		if real is None: real = self.isReal

		# Choose the right function
		fftfunc = {
				(True, False): fft.rfft,
				(True, True): fft.irfft,
				(False, False): fft.fft,
				(False, True): fft.ifft
			}[(real, inverse)]

		try:
			# Find overlap between the data and output windows
			ostart, istart, wlen = cutil.overlap(window, self.datawin)
		except TypeError:
			# There is no overlap, the signal window and its FFT are 0
			# Take an empty transform to get the right output form
			return fftfunc(np.array([], dtype=self.dtype), n=window[1])

		oend, iend = ostart + wlen, istart + wlen

		if ostart < 1:
			# If the signal window starts before sample 1,
			# no need exists to pre-pad the FFT input
			sig = self._data[istart:iend]
		else:
			# Otherwise, the signal window starts with some zeros
			# An intermediate array must be created
			sig = np.zeros((oend,), dtype=self.dtype)
			sig[ostart:oend] = self._data[istart:iend]

		return fftfunc(sig, n=window[1])


	def aligned(self, ref, osamp=1, allowneg=False):
		'''
		Return a copy of the signal aligned to the reference
		habis.sigtools.Waveform ref by calling self.shift() with the
		output of self.delay(ref, osamp, allowneg).
		'''
		return self.shift(self.delay(ref, osamp, allowneg))


	def shift(self, d):
		'''
		Return a copy of the waveform cyclically shifted by a number of
		(optionally fractional) samples d using a spectral multiplier.
		Positive shifts correspond to delays (i.e., the signal moves
		right).

		Fourier transforms are always computed over the entire signal.
		'''
		# Determine whether the FFT will be R2C or C2C
		r2c = self.isReal
		ifftfunc = fft.irfft if r2c else fft.ifft

		n = self.nsamp

		fsig = self.fft(window=(0, n), real=r2c)
		nsig = len(fsig)

		# Build the spectral indices
		kidx = np.arange(nsig)
		# Correct negative frequencies in a C2C transform
		if not r2c: kidx[(nsig + 1) / 2:] -= nsig

		# Build the shift operator
		sh = np.exp(-2j * math.pi * d * kidx / n)
		# Correct the Nyquist frequency term for conjugate symmetry
		if n % 2 == 0: sh[n / 2] = np.real(sh[n / 2])

		ssig = ifftfunc(fsig * sh, n).astype(self.dtype)

		# Return a copy of the shifted signal
		return Waveform(self.nsamp, ssig, 0)


	def delay(self, ref, osamp=1, allowneg=False):
		'''
		Find the delay that maximizes the cross-correlation between
		this instance and another habis.sigtools.Waveform ref.

		Both signals will be interpolated by at least a factor osamp,
		if provided, to yield fractional-sample delays.

		If allowneg is True, the return value will be a tuple (d, s),
		where d is the delay that maximizes the absolute value of the
		cross-correlation, and s is the sign (-1 or 1, integer) of the
		cross-correlation at this point.
		'''
		if osamp < 1:
			raise ValueError ('Oversampling factor must be at least unity')

		sstart, slength = self.datawin
		rstart, rlength = ref.datawin

		# If one of the signals is explicitly 0, the delay is 0
		if slength == 0 or rlength == 0: return 0

		# Use an R2C DFT if both signals are real, otherwise complex
		r2c = self.isReal and ref.isReal
		ifftfunc = (fft.irfft if r2c else fft.ifft)

		# Find the size of FFT needed to represent the correlation
		npad = cutil.ceilpow2(slength + rlength)
		# Find the next power of 2 needed to capture the desired oversampling
		nint = cutil.ceilpow2(osamp * npad)

		# Compute the interpolated cross-correlation
		# Always take the real part to look for a maximum
		csig = ifftfunc(self.fft((sstart, npad), r2c) *
				np.conj(ref.fft((rstart, npad), r2c)), nint).real

		# Find the point of maximal correlation
		if allowneg:
			t = np.argmax(np.abs(csig))
			# Determine the sign of the correlation
			s = int(math.copysign(1., csig[t]))
		else: t = np.argmax(csig)

		# Unwrap negative values and scale by the oversampling rate
		if t >= nint / 2: t = t - nint
		t *= npad / float(nint)

		# Adjust the delay to account for different data window positions
		t += (sstart - rstart)

		return ((t, s) if allowneg else t)


	def envpeaks(self, thresh=3., twin=None):
		'''
		Return a (start, length) tuple to specify the window for each
		peak identified in the envelope() of the waveform. A peak is a
		region that exceeds the mean envelope value by at least thresh
		standard deviations.

		If twin is None, the mean envelope value and standard deviation
		used for thresholding are computed using the full signal.
		Otherwise, twin should be a (start, length) tuple that
		specifies the window over which the the mean and standard
		deviation will be computed.
		'''
		if thresh < 0:
			raise ValueError('Argument thresh must be positive')

		if twin is None: twin = (0, self.nsamp)

		# Compute the envelope of the entire signal
		env = self.envelope()
		# Compute the threshold parameters
		estat = env[twin[0]:twin[0] + twin[1]]
		emu = np.mean(estat)
		esd = np.std(estat)

		# Find all indices that exceed the threshold
		peaks = (env > (thresh * esd + emu)).nonzero()[0]
		# Group continguous indices
		groups = [map(itemgetter(1), g)
				for k, g in groupby(enumerate(peaks), lambda(i, x): i - x)]
		return [(g[0], g[-1] + 1) for g in groups]


	def bandpass(self, start, end, tails=None, crop=False):
		'''
		Perform a bandpass operation that zeros all frequencies below
		the frequency bin at index start and above (and including) the
		frequency bin at index end. Only positive frequencies may be
		specified.

		If tails is provided, it should be a 1-D array of length N that
		will modify the frequencies of sig according to the formula

			fsig[start:start+N/2] *= tails[:N/2],
			fsig[end-N/2:end] *= tails[-N/2:],

		where fsig = rfft(sig). If the signal is complex (and,
		therefore, a C2C DFT is used, negative frequencies are
		similarly modified.

		If crop is True, the data window of the output filtered
		waveform will be cropped to the original data window.
		Otherwise, the entire duration of the filtered waveform will be
		preserved in the output.
		'''
		# Check the window for sanity
		if start >= end:
			raise ValueError('Starting index should be less than ending index')

		# Check the tail for sanity
		if tails is not None and len(tails) > (end - start):
			raise ValueError('Single-side tail should not exceed half window width')

		r2c = self.isReal
		n = self.nsamp

		# Transform the signal and apply the frequency window
		fsig = self.fft(window=(0, n), real=r2c)

		if r2c:
			fmax = len(fsig)
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
		if tails is not None:
			ltails = len(tails) / 2
			fsig[start:start+ltails] *= tails[:ltails]
			fsig[end-ltails:end] *= tails[-ltails:]

			if not r2c:
				tails = np.conj(tails)
				fsig[n-start:n-start-ltails:-1] *= tails[:ltails]
				fsig[n-end+ltails:n-end:-1] *= tails[-ltails:]

		ifftfunc = fft.irfft if r2c else fft.ifft
		bsig = ifftfunc(fsig, n).astype(self.dtype)

		if crop:
			start, length = self.datawin
			return Waveform(self.nsamp, bsig[start:start+length], start)
		else:
			return Waveform(self.nsamp, bsig, 0)


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
