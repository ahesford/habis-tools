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
	sig = np.squeeze(sig)
	if ndim is not None and sig.ndim != ndim:
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


def delay(sig, ref, osamp=1, restrict=None):
	'''
	Determine the delay of a signal sig that maximizes the
	cross-correlation between sig and a reference signal ref.

	The signal and reference will be interpolated by at least a factor
	osamp, if provided, to yield fractional-sample delays.

	If restrict is provided and not None, it should be a sequence (a, b)
	that specifies the range of samples (at the original sampling rate)
	over which cross-correlation should be maximized.

	Both sig and ref should be 1-D arrays or column vectors.
	'''
	if osamp < 1:
		raise ValueError ('Oversampling factor must be at least unity')

	# Ensure dimensional compatibility of the input arrays
	sig = dimcompat(sig, 1)
	ref = dimcompat(ref, 1)

	# Find the size of FFT needed to represent the correlation
	npad = cutil.ceilpow2(len(sig) + len(ref))
	sfft = fft.fft(sig, npad)
	rfft = fft.fft(ref, npad)

	# Find the next power of 2 needed to capture the desired oversampling
	nint = cutil.ceilpow2(osamp * npad)
	cfft = np.zeros((nint,), dtype=sfft.dtype)
	# Copy the convolution FFT into the padded array
	halfpad = npad / 2
	cfft[:halfpad] = sfft[:halfpad] * np.conj(rfft[:halfpad])
	cfft[-halfpad:] = sfft[-halfpad:] * np.conj(rfft[-halfpad:])
	csig = fft.ifft(cfft)
	# Find the delay, restricting the range if desired
	if restrict is not None:
		start, end = [int(r * nint / npad) for r in restrict[:2]]
		t = np.argmax(csig.real[start:end]) + start
	else: t = np.argmax(csig.real)
	# Unwrap negative values
	if t >= nint / 2: t = t - nint
	# Scale the index by the real oversampling rate
	return t * npad / float(nint)


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


def shifter(sig, d, n=None):
	'''
	Shift the signal sig by a number of (optionally fractional) samples d
	using a spectral multiplier. If n is provided, the signal will be
	padded (or truncated) to length n before its FFT is computed. The
	output signal retains its modified length.

	The signal should be a 1-D array.
	'''
	sig = dimcompat(sig, 1)
	fsig = fft.fft(sig, n)
	nsig = len(fsig)
	kidx = fft.fftshift(np.arange(-nsig/2, nsig/2))
	sh = np.exp(2j * math.pi * d * kidx / nsig)
	ssig = fft.ifft(fsig * sh)
	if not np.issubdtype(sig.dtype, np.complexfloating):
		ssig = ssig.real
	return ssig.astype(sig.dtype)


def envpeaks(sig, thresh=3., nwin=None):
	'''
	Return a (start, end) tuple to specify the slice [start:end] for each
	peak identified in the envelope of the signal sig. The envelope is the
	magnitude of the analytic signal computed by scipy.signal.hilbert. A
	peak is a region that exceeds the mean envelope value by at least
	thresh standard deviations.

	If nwin is None, the mean envelope value and standard deviation used
	for thresholding are computed using the full signal. Otherwise, if nwin
	is a slice object, the mean and standard deviation will be computed
	using the specified slice of the envelope. If nwin is an integer, the
	slice is assumed to start at 0 and end at nwin.

	The signal should be a 1-D real array.
	'''
	sig = dimcompat(sig, 1)
	if thresh < 0:
		raise ValueError('Argument thresh must be positive')
	# Ensure that the slice value is appropriate
	if nwin is None or isinstance(nwin, int):
		nwin = slice(nwin)
	elif not isinstance(nwin, slice):
		raise TypeError('Argument nwin must be None, integer or slice')

	# Compute the Hilbert transform
	env = np.abs(hilbert(sig))
	emu = np.mean(env[nwin])
	esd = np.std(env[nwin])

	# Find all indices that exceed the threshold
	peaks = (env > (thresh * esd + emu)).nonzero()[0]
	# Group continguous indices
	groups = [map(itemgetter(1), g)
			for k, g in groupby(enumerate(peaks), lambda(i, x): i - x)]
	return [(g[0], g[-1] + 1) for g in groups]


def bandpass(sig, start, end, tails=None):
	'''
	Given a time-domain signal sig, perform a bandpass operation that zeros
	all frequencies below the frequency bin at index start and above (and
	including) the frequency bin at index end. A complex FFT is assumed, so
	the filter will be applied to positive and negative indices.

	If tails is provided, it should be a 1-D array of length N that will
	modify the frequencies of sig according to the formula

		fsig[start:start+N/2] *= tails[:N/2],
		fsig[end-N/2:end] *= tails[-N/2:],
		fsig[-start-N/2:-start] *= conj(reversed(tails[:N/2])),
		fsig[-end:-end+N/2] *= conj(reversed(tails[-N/2:])),

	where fsig = fft(sig).

	Both sig and tails should be 1-D arrays.
	'''
	# Ensure that the input is 1-D compatible
	sig = dimcompat(sig, 1)
	# Find the maximum positive frequency and the minimum negative frequency
	lsig = len(sig)
	fmax = int(lsig - 1) / 2
	fmin = -int(lsig / 2)

	# Check the window for sanity
	if start >= end:
		raise ValueError('Starting index should be less than ending index')
	if start < 0 or start >= fmax:
		raise ValueError('Starting index should be positive but less than maximum frequency')
	if end < 0 or end > fmax:
		raise ValueError('Ending index should be positive but not exceed maximum frequency')

	# Check the tail for sanity
	if tails is not None and len(tails) > (end - start):
		raise ValueError('Single-side tail should not exceed half window width')

	# Transform the signal and apply the window
	fsig = fft.fft(sig)
	# Positive frequency window
	fsig[:start] = 0
	fsig[end:fmax+1] = 0
	# Negative frequency window
	fsig[lsig+fmin:lsig-end+1] = 0
	fsig[lsig-start+1:] = 0

	# Apply the tails
	if tails is not None:
		ltails = len(tails) / 2
		fsig[start:start+ltails] *= tails[:ltails]
		fsig[end-ltails:end] *= tails[-ltails:]
		fsig[lsig-start:lsig-start-ltails:-1] *= np.conj(tails[:ltails])
		fsig[lsig-end+ltails:lsig-end:-1] *= np.conj(tails[-ltails:])

	# If the input was real, ensure that the output is real too
	bsig = fft.ifft(fsig)
	if not np.issubdtype(sig.dtype, np.complexfloating):
		bsig = bsig.real
	return bsig.astype(sig.dtype)
