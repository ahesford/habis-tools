'''
Routines for manipulating HABIS signals.
'''

import numpy as np, math
from numpy import fft
from scipy.signal import hilbert
from operator import itemgetter
from itertools import groupby

from pyajh import cutil

def delay(sig, ref, osamp=1):
	'''
	Determine the delay of a signal sig that maximizes the
	cross-correlation between sig and a reference signal ref.

	The signal and reference will be interpolated by at least a factor
	osamp, if provided, to yield fractional-sample delays.

	Both sig and ref should be 1-D arrays or column vectors.

	Correlations are done using single precision.
	'''
	if osamp < 1:
		raise ValueError ('Oversampling factor must be at least unity')
	# Find the size of FFT needed to represent the correlation
	npad = cutil.ceilpow2(len(sig) + len(ref))
	sfft = fft.fft(sig, npad)
	rfft = fft.fft(ref, npad)

	# Find the next power of 2 needed to capture the desired oversampling
	nint = cutil.ceilpow2(osamp * npad)
	cfft = np.zeros((nint,), dtype=np.complex64)
	# Copy the convolution FFT into the padded array
	halfpad = npad / 2
	cfft[:halfpad] = sfft[:halfpad] * np.conj(rfft[:halfpad])
	cfft[-halfpad:] = sfft[-halfpad:] * np.conj(rfft[-halfpad:])
	# Find the delay and handle wrapping properly
	t = np.argmax(fft.ifft(cfft).real)
	if t >= nint / 2: t = t - nint
	return t * npad / float(nint)


def normalize(sig, env):
	'''
	Multiply the spectral response of the signal sig by the provided
	envelope env. The signal is padded (or truncated) to match the
	length of env before its FFT is computed. If the signal is padded, the
	output signal will be truncated to the length of the input signal.

	Both sig and env should be 1-D arrays or column vectors.
	'''
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
	fsig = fft.fft(sig, n)
	nsig = len(fsig)
	kidx = fft.fftshift(np.arange(-nsig/2, nsig/2))
	sh = np.exp(2j * math.pi * d * kidx / nsig)
	return fft.ifft(fsig * sh)


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
