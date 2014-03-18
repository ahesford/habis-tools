'''
Routines for manipulating HABIS signals.
'''

import numpy as np, math
from numpy import fft
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
	Compute a spectral envelope to approximately convert the spectral
	magnitude of the input signal to the desired shape given in ref. The
	signal is padded (or truncated) to match the length of ref. The output
	envelope retains the modified length.

	Wherever the spectral magnitude of sig falls below the specified
	threshold thresh, the envelope will have zero amplitude.

	Both sig and env should be 1-D arrays or column vectors.
	'''
	fsig = fft.fft(sig, len(ref))
	# Eliminate zero to avoid complaints
	fsiga = (np.abs(fsig) > thresh).choose(1., abs(fsig))
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
