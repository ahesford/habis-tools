'''
Routines for efficiently computing short-time Fourier transforms of 1-D
signals, and their inverses. The interface is simpler than scipy.signal.stft
and can optionally use PyFFTW without monkey-patching scipy.fftpack.
'''

# Copyright (c) 2018 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np

from pycwp.stats import rolling_window

def stft(x, win, nfft=None, fftw=None, return_plan=False):
	'''
	Compute D, a short-time Fourier transform of the 1-D signal x such that 
	D[n,k] is the value of the DFT for frequency bin k in time frame n. The
	method pycwp.stats.rolling_window is used to efficiently subdivide the
	signal into overlapping windows.

	The STFT window size is given by the length of win, which should also
	be a 1-D Numpy array or compatible sequence. The hop length is always
	unity, so adjacent time frames overlap by len(win) - 1 samples.

	If nfft, the length of the DFT, is None, a default value that is the
	smallest regular number at least as large as len(win) will be used.
	Otherwise, the value of nfft will be used as specified; a ValueError
	will be raised when nfft is not at least as large as len(win).

	If neither x nor win are complex, an R2C DFT will be used, which means
	the number of frequency bins in the output will be (nfft // 2) + 1;
	otherwise, a fully complex DFT will be used.

	If fftw is not None and the PyFFTW module is available, it should be a
	pyfftw.FFTW object that will be used to perform a forward FFT of shape
	(len(x) - len(win) + 1, nfft). If fftw is compatible (which means that
	fftw.input_dtype is complex iff x is complex or win is complex, the
	input shapes are compatible and the transform occurs along axis 1), the
	DFT is computed by calling fftw(input), where input is the rolling-
	window representation of x multiplied by the window function. Note
	that this will overwrite the input of the FFTW object.

	If fftw is not compatible with the input, fftw is None or the PyFFTW
	module cannot be imported, a new FFT will be planned (if pyfftw is
	available) or numpy.fft will be used instead.

	If return_plan is True, the return value will be the STFT and the
	PyFFTW plan used for the Fourier transform (this is useful for repeated
	calls with different inputs), or None if PyFFTW was not used. If
	return_plan is False, the return value is just the STFT array.
	'''
	x = np.asarray(x).squeeze()
	win = np.asarray(win).squeeze()

	if x.ndim != 1 or win.ndim != 1:
		raise ValueError('Inputs x and win must be 1-D or compatible')

	lwin = len(win)

	if nfft is None:
		try: from scipy.fftpack.helper import next_fast_len
		except ImportError: nfft = lwin
		else: nfft = next_fast_len(lwin)

	if nfft < lwin:
		raise ValueError('Value of nfft must be no smaller than len(win)')

	# Find the input data type
	xw = rolling_window(x, lwin) * win[np.newaxis,:]
	r2c = not np.issubdtype(xw.dtype, np.complexfloating)
	fftname = 'rfft' if r2c else 'fft'

	try: import pyfftw
	except ImportError: use_pyfftw = False
	else: use_pyfftw = True

	if not use_pyfftw:
		# Use Numpy for FFTs
		fftfunc = getattr(np.fft, fftname)
		out = fftfunc(xw, nfft, 1)
		if return_plan: return out, None
		else: return out

	# Check validity of existing plan
	try:
		if fftw is None or fftw.axes != (1,):
			raise ValueError
		if r2c == np.issubdtype(fftw.input_dtype, np.complexfloating):
			raise ValueError
		out = fftw(xw)
	except ValueError:
		builder = getattr(pyfftw.builders, fftname)
		fftw = builder(xw, nfft, axis=1)
		out = fftw()

	out = out.copy()
	if return_plan: return out, fftw
	else: return out


def istft(y, win, nfft=None, ifftw=None, return_plan=False):
	'''
	From y, a short-time Fourier transform as a 2-D array with time-frame
	indices along the first axis and frequency-bin indices along the
	second, synthesize the corresponding signal x.

	The STFT window must be a 1-D Numpy array or compatible sequence and
	should match the forward-transform window. The hop length is always
	unity, so adjacent time frames overlap by len(win) - 1 samples.

	If nfft, the length of the DFT, is None, a default value that is the
	smallest regular number at least as large as len(win) will be used.
	Otherwise, the value of nfft will be used as specified; a ValueError
	will be raised when nfft is not at least as large as len(win).

	If y.shape[1] is equal to nfft, a C2C FFT is assumed; if y.shape[1] is
	equal to (nfft // 2) + 1, an R2C FFT is assumed; otherwise, a
	ValueError will be raised.

	If ifftw is not None and the PyFFTW module is available, it should be a
	pyfftw.FFTW object that will be used to perform an inverse transform of
	shape (len(x) - len(win) + 1, nfft). If ifftw is compatible (which
	means that the input shapes are compatible and the transform occurs
	along axis 1), the DFT is computed by calling ifftw(y). Note that this
	will overwrite the input of the FFTW object.

	If ifftw is not compatible with the input, ifftw is None or the PyFFTW
	module cannot be imported, a new inverse FFT will be planned (if pyfftw
	is available) or numpy.fft will be used instead.

	If return_plan is True, the return value will be the signal and the
	PyFFTW plan used for the inverse Fourier transform (this is useful for
	repeated calls with different inputs), or None if PyFFTW was not used.
	If return_plan is False, the return value is the synthesized signal.
	'''
	y = np.asarray(y).squeeze()
	win = np.asarray(win).squeeze()

	if y.ndim != 2 or win.ndim != 1:
		raise ValueError('Input x must be 2-D, win must be 1-D (or compatible)')

	lwin = len(win)

	if nfft is None:
		try: from scipy.fftpack.helper import next_fast_len
		except ImportError: nfft = lwin
		else: nfft = next_fast_len(lwin)

	if nfft < lwin:
		raise ValueError('Value of nfft must be no smaller than len(win)')

	if y.shape[1] == nfft: r2c = False
	elif y.shape[1] == nfft // 2 + 1: r2c = True
	else: raise ValueError('Last axis of y incompatible with nfft specification')

	# Find the input data type
	ifftname = 'irfft' if r2c else 'ifft'

	try: import pyfftw
	except ImportError: use_pyfftw = False
	else: use_pyfftw = True

	if not use_pyfftw:
		# Use Numpy for FFTs
		ifftfunc = getattr(np.fft, ifftname)
		out = ifftfunc(y, nfft, 1)
		ifftw = None
		if return_plan: return out, None
		else: return out
	else:
		# Check validity of existing plan
		try:
			if ifftw is None or ifftw.axes != (1,):
				raise ValueError
			if r2c == np.issubdtype(ifftw.output_dtype, np.complexfloating):
				raise ValueError
			out = ifftw(y)
		except ValueError:
			builder = getattr(pyfftw.builders, ifftname)
			ifftw = builder(y, nfft, axis=1)
			out = ifftw()

	# Apply STFT window to decomposed output
	out = out[:,:lwin] * win[np.newaxis,:]

	# Synthesize and normalize the output
	x = np.zeros((out.shape[0] + out.shape[1] - 1,), dtype=out.dtype)
	x[:out.shape[0]] = out[:,0]
	for i, row in enumerate(out.T[1:], 1): x[i:i+out.shape[0]] += row
	x /= np.sum(np.abs(win)**2)

	if return_plan: return x, ifftw
	else: return x
