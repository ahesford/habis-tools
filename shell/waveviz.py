#!/usr/bin/env python

import numpy as np, getopt, sys, os

from math import sqrt

from argparse import ArgumentParser

from scipy.signal import hilbert

import itertools

import progressbar

from collections import defaultdict

from habis.habiconf import matchfiles
from habis.sigtools import Waveform, Window
from habis.formats import loadkeymat, WaveformSet

def plotframes(output, waves, atimes, dwin=None,
		equalize=False, cthresh=None, bitrate=-1, one_sided=False):
	'''
	Prepare, using the ffmpeg writer in matplotlib.animation, a video with
	the given bitrate in which each frame depicts aligned waveforms
	received by a common element. The argument waves should be a mapping
	from transmit-receive pairs to a list of Waveform objects, prealigned
	if alignment is desired.

	The argument atimes should be a mapping from element index to a list of
	at least one arrival time. If the mapping contains at least one arrival
	time for a given element index, the first arrival time for the element
	will be plotted along with the corresponding waveforms.

	If dwin = (start, win) is specified, it defines an absolute window
	(when atimes is None) or relative window (when atimes is defined) over
	which the waveforms (and arrival times) will be plotted. In the
	relative mode, the actual plot window starts at

		start + min(atimes[pair][0] for pair in cpairs)

	and ends at

		end + max(atimes[pair][0] for pair in cpairs),

	where cpairs is the list of comment keys in atimes and waves.

	If dwin is None, the window will be chosen to encompass all data
	windows in the waves map.

	If equalize is True (or a value greater than 0), each wave group will
	be equalized by calling eqwavegrps with a value (equalize > 1) for the
	"individual" argument to the function. The equalization is done after
	dwin is applied to each waveform.

	The value of cthresh, if not None, specifies the number of standard
	deviations above the mean of the peak amplitudes of all displayed
	waveforms that establishes the upper limit on the vertical scale. For
	the corresponding value CMAX = <PEAK MEAN> + cthresh * <PEAK STD>, the
	vertical scale will range from -CMAX to CMAX. If cthresh is None, CMAX
	will assume the largest peak amplitude displayed in the video.

	If one_sided is True, the vertical scale will run from 0 to CMAX
	instead of -CMAX to CMAX.
	'''
	import matplotlib as mpl
	mpl.use('agg')
	import matplotlib.pyplot as plt
	import matplotlib.animation as ani

	# Ensure all data sets are equally sized
	wvit = iter(waves.values())
	try: nsets = len(next(wvit))
	except StopIteration: nsets = 0
	if any(len(v) != nsets for v in wvit):
		raise ValueError('All waveform lists must be equally sized')

	if nsets < 2 and atimes is not None:
		atit = iter(atimes.values())
		try: ntimes = len(next(atit))
		except StopIteration: ntimes = 0
		if any(len(v) != ntimes for v in atit):
			raise ValueError('All arrival time lists must be equally sized')
	elif atimes is not None:
		ntimes = 1
	else: ntimes = 0

	# Prepare the axes for a 1080p frame
	fig = plt.figure()
	fig.set_dpi(80)
	fdpi = float(fig.get_dpi())
	fig.set_size_inches(1920. / fdpi, 1080. / fdpi)
	fig.subplots_adjust(left=0.1, right=0.975, bottom=0.1, top=0.9)

	# Grab the axes
	ax = ax = fig.add_subplot(111)

	# Prepare the video writer
	try: ffwr = ani.writers['ffmpeg']
	except KeyError:
		raise KeyError('The ffmpeg animation writer is required for video creation')

	# Configure the writer (let ffmpeg decide the bitrate)
	metadata = dict(title='Waveform analysis video', artist='waveviz.py')
	writer = ffwr(fps=5, bitrate=bitrate, metadata=metadata)

	if dwin is None:
		# With no data window, show the entire data range
		dstart = min(w.datawin.start for v in waves.values() for w in v)
		dend = max(w.datawin.end for v in waves.values() for w in v)
		dwin = Window(dstart, end=dend)
	else:
		if atimes is not None:
			# The window is relative to the arrival-time range
			cpairs = set(waves.keys()).intersection(iter(atimes.keys()))
			if nsets > 1:
				# With more than one waveform, first time is relevant
				dstart = min(atimes[pair][0] for pair in cpairs)
				dend = max(atimes[pair][0] for pair in cpairs)
			else:
				# With a single waveform, all times will be shown
				dstart = min(min(atimes[pair]) for pair in cpairs)
				dend = max(max(atimes[pair]) for pair in cpairs)
			dwin = Window(max(0, int(dstart + dwin[0])), end=int(dend + dwin[1]))
		else:
			dwin = Window(dwin[0], end=dwin[1])

	# Clip the waveforms to the common data window
	waves = { k: [ w.window(dwin) for w in v ] for k, v in waves.items() }

	# Equalize the waveforms if desired
	if equalize:
		if equalize > 1: print('Equalizing waveforms individually')
		waves = eqwavegrps(waves, equalize > 1)

	# Set the amplitude limits
	pkamps = [ w.envelope().extremum()[0]
			for v in waves.values() for w in v ]

	if cthresh is None: vmax = np.max(pkamps)
	else: vmax = np.mean(pkamps) + cthresh * np.std(pkamps)

	if not one_sided: vmin = -vmax
	else: vmin = 0

	# Build the common time axis
	taxis = np.arange(dwin.start, dwin.end)

	print(f'Waveform count: {nsets}; arrival-time count: {ntimes}')
	print(f'Display frame is [{dwin.start}, {dwin.end}, {vmin:g}, {vmax:g}]')

	# Create the frames and write the video
	with writer.saving(fig, output, fig.get_dpi()):
		# Create the empty plot for efficiency
		lines = ax.plot(*[[] for i in range(2 * nsets)])

		# Create empty plots for arrival times, with vertical lines
		cycler = mpl.rcParams['axes.prop_cycle']()

		# Skip past the number of colors already used
		for i in range(nsets): next(cycler)

		# Set lines for arrival times, with colors in the cycle
		apoints, alines = [ ], [ ]
		for i in range(ntimes):
			color = next(cycler)['color']
			apt = ax.plot([], [], linestyle='', marker='o', color=color)
			apoints.extend(apt)
			aln = ax.axvline(color=color, linestyle='--')
			alines.append(aln)

		ax.axis([taxis[0], taxis[-1], vmin, vmax])
		ax.set_xlabel('Time, samples', fontsize=14)
		ax.set_ylabel('Amplitude', fontsize=14)
		ax.grid(True)

		bar = progressbar.ProgressBar(max_value=len(waves))

		for i, (pair, wlist) in enumerate(sorted(waves.items())):
			# Update the line data
			for l, w in zip(lines, wlist):
				l.set_data(taxis, w.getsignal(dwin))

			# Plot an arrival time, if possible
			try:
				atelts = [int(v) for v in atimes[pair][:ntimes]]
			except (KeyError, TypeError, IndexError):
				for apoint in apoints: apoint.set_visible(False)
				for aline in alines: aline.set_visible(False)
			else:
				for apt, aln, ate in zip(apoints, alines, atelts):
					apt.set_data([ate], [wlist[0][ate]])
					apt.set_visible(True)
					aln.set_xdata([ate, ate])
					aln.set_visible(True)

			ax.set_title(f'Waveform {pair}', fontsize=14)

			# Capture the frame
			writer.grab_frame()
			bar.update(i)

		bar.update(len(waves))


def plotwaves(output, waves, atimes=None, mtime=None,
		dwin=None, log=False, cthresh=None, one_sided=False):
	'''
	Plot, into the image file output, the habis.sigtools.Waveform objects
	mapped (by index) in waves, with temporal variations along the vertical
	axis. The index along the horizontal axis is into sorted(waves) and is
	not guaranteed to correspond to the index in the mapping.

	If atimes is not None, it should map indices to waveform arrival times.
	A subplot will show these arrival times on the same horizontal axis as
	the waveform image. Elements in waves that do not exist in atimes will
	be replaced with NaN when plotting arrival times.

	If mtime is not None, it should be the mean arrival time use to align
	the waveforms. In this case, the time will be printed in the title of
	the arrival-time plot.

	The waves are cropped to the specified data window prior to plotting.
	If dwin is None, the smallest data window that encompasses all plotted
	signals will be used.

	If log is True, the plot will display log magnitudes rather than linear
	waveforms. The maximum color value will always be the peak amplitude,
	and cthresh, if not None, should be a negative value that specifies the
	minimum resolvable magnitude in dB down from the maximum or a positive
	value that specifies the maximum resolvable magnitude in dB above an
	estimate of the minimum noise level over a 200-sample sliding window.

	If log is False, the color range will clip at cthresh standard
	deviations above the mean peak amplitude over all signals. If cthresh
	is None, the narrowest range that avoids clipping will be selected.

	When one_sided is False, the color scale will be symmetric about zero
	based on the maximum value determined by cthresh. When one_sided is
	True (which is not possible when log is True), the low end of the color
	scale will be zero.
	'''
	import matplotlib as mpl
	mpl.use('pdf')
	import matplotlib.pyplot as plt
	from matplotlib import cm

	# Split the mapping in indexed order
	widx, waves = zip(*sorted(waves.items()))

	if log and one_sided:
		raise ValueError('Cannot have both log==True and one_sided==True')

	# Pull the relevant arrival times for a subplot
	if atimes is not None:
		atimes = [ atimes.get(k, float('nan')) for k in widx ]

	if dwin is None:
		dstart = min(w.datawin.start for w in waves)
		dend = max(w.datawin.end for w in waves)
		dwin = Window(dstart, end=dend)

	# Prepare the axes
	fig = plt.figure()
	fig.set_dpi(600)
	fdpi = float(fig.get_dpi())

	# Figure the figure size so each value gets at least 1 pixel
	spwfrac = fig.subplotpars.right - fig.subplotpars.left
	sphfrac = (fig.subplotpars.top - fig.subplotpars.bottom) / 2.
	if atimes is not None:
		# Account for an arrival-time subplot
		sphfrac -= fig.subplotpars.hspace / 2.
	wfig = max(12., np.ceil(float(len(waves)) / spwfrac / fdpi))
	hfig = max(np.ceil(float(dwin[1]) / sphfrac / fdpi), wfig / 3.)
	fig.set_size_inches(wfig, hfig)

	# Add axes to contain the plots
	if atimes is None:
		ax = [fig.add_subplot(111)]
	else:
		ax = [fig.add_subplot(211)]
		ax.append(fig.add_subplot(212, sharex=ax[0]))

	# Pull the waveforms and determine the color range
	img = np.array([w.getsignal(dwin) for w in waves])

	if not log:
		pkamps = np.max(np.abs(hilbert(img, axis=1)), axis=1)

		if cthresh is None: cmax = np.max(pkamps)
		else: cmax = np.mean(pkamps) + cthresh * np.std(pkamps)

		if not one_sided:
			clim = [-cmax, cmax]
			cmap = cm.RdBu
		else:
			clim = [0, cmax]
			cmap = cm.Reds
	else:
		# Compute signal image to log magnitude
		img = np.abs(hilbert(img, axis=1))
		# Clip approximately-zero values
		imax = np.max(img)
		imin = np.min(img[np.nonzero(img)])
		img = np.log10(np.clip(img, imin, imax))

		pkval = np.max(img)

		if cthresh is None:
			clim = [np.min(img), pkval]
		elif cthresh < 0:
			clim = [pkval + cthresh / 20., pkval]
		else:
			# Estimate noise levels for image
			nlev = min(w.noisefloor(200) for w in waves)
			# If no noise level was found, default to down-from-peak
			# Otherwise, convert dB to simple log values
			if np.isinf(nlev): clim = [pkval - cthresh / 20., pkval]
			else: clim = [ nlev / 20., (nlev + cthresh) / 20. ]

		cmap = cm.Reds

	# Shift extent by half a pixel so grid lines are centered on samples
	extent = [-0.5, img.shape[0] - 0.5, dwin[0] - 0.5, dwin[0] + dwin[1] - 0.5]

	# Plot the waveform image
	ax[0].imshow(img.T, vmin=clim[0], vmax=clim[1], cmap=cmap,
			interpolation='nearest', origin='lower', extent=extent)
	ax[0].grid(True)
	ax[0].set_aspect('auto')
	ax[0].set_ylabel('Time, samples', fontsize=16)
	if atimes is not None:
		title = 'Waveforms aligned to mean arrival time'
		if mtime: title += f' ({mtime} samples)'
	else:
		title = 'Waveforms with natural alignment'
	ax[0].set_title(title + (' (log magnitude)' if log else ''), fontsize=16)

	if atimes is not None:
		# Plot the arrival-time image
		ax[1].plot(atimes, linewidth=0.5)
		ax[1].grid(True)
		ax[1].set_xlabel('Waveform index', fontsize=16)
		ax[1].set_ylabel('Time, samples', fontsize=16)
		ax[1].set_title('Waveform arrival times', fontsize=16)
		plt.setp(ax[0].get_xticklabels(), visible=False)

	# Ensure at least 10 x ticks exist
	ax[0].set_xlim(0, img.shape[0])
	if len(ax[0].get_xticks()) < 10:
		ax[0].set_xticks(list(range(0, img.shape[0] + 1, img.shape[0] // 10)))

	# Save the image
	fig.savefig(output, bbox_inches='tight')


def getatimes(atarg, freq=1, scalar=True, cols=None):
	'''
	Given a list of files or globs, try to open arrival-time maps matching
	the globs with habis.formats.loadkeymat() and pull the columns
	specified in the sequence cols. If cols is None, all columns will be
	picked. Keys of each map should be transmit-receive pairs (t, r).

	Files are loaded in lexical order. If the same key is present in
	multiple files, the values for that key will a concatenation of the
	values for individual files (each considered as a list) that preserves
	the lexical ordering.

	If the lengths of value lists for keys in the composite arrivla-time
	map, only those keys with maximum-length values will be retained.

	The times are scaled by the frequency to convert the times to samples.

	If scalar is True, values in the returned map will be scalars if a
	single column is pulled. Otherwise, the returned values will always be
	arrays.
	'''
	# Try to load the files one-by-one
	atfiles = sorted(matchfiles(atarg, forcematch=True))

	# Concatenate values to accommodate repeat keys, track max column count
	ncols = 0
	atmap = defaultdict(list)
	for atfile in atfiles:
		for k, v in loadkeymat(atfile, scalar=False, nkeys=2).items():
			atmap[k].extend(vv for vv in v)
			ncols = max(ncols, len(atmap[k]))

	if cols is None:
		acols = list(range(ncols))
	else:
		acols = cols
		print(f'Using columns {acols} from arrival-time records')

	if scalar:
		if len(acols) != 1:
			raise ValueError('Scalar arrival-time map requires a single column specification')
		acols = acols[0]

	return { k: freq * np.array(v)[acols]
			for k, v in atmap.items() if len(v) == ncols }


def shiftgrps(wavegrps, atimes, suppress=False):
	'''
	In a mapping wavegrps as returned by getwavegrps, shift each waveform
	wavegrps[t,r][i] by the difference atimes[t,r][0] - atimes[t,r][i]. If
	a list atimes[t,r] cannot be found, no shift will be performed for that
	(t, r) pair. If suppress is True, any (t,r) pair in wavegrps without a
	corresponding list atimes[t,r] will be excluded from the output.

	If the length of atimes[t,r] is unity or the length of wavegrps[t,r] is
	unity, no shift will be performed, but the list of waveforms will be
	included in the output regardless of the value of suppress.

	If both atimes[t,r] and waevgrps[t,r] have non-unit length but the
	lengths do not match,an IndexError will be raised.
	'''
	output = { }
	for (t,r) in wavegrps.keys():
		# Pull the waveform list
		waves = wavegrps[t,r]

		# Pull the arrival-time list, if possible
		try: atlist = atimes[t,r]
		except KeyError:
			if not suppress: output[t,r] = waves
			continue

		# With a single time or waveform, no shifting is performed
		if len(atlist) == 1 or len(waves) == 1:
			output[t,r] = waves
			continue

		if len(atlist) != len(waves):
			raise IndexError('Length of arrival-time list does not match '
					f'length of wave-group list for pair {(t,r)}')

		# Build the new list of shifted waves
		output[t,r] = [ waves[0] ]
		output[t,r].extend(wf.shift(atlist[0] - atv)
					for wf, atv in zip(waves[1:], atlist[1:]))

	return output


def eqwavegrps(wavegrps, individual=False):
	'''
	In a mapping wavegrps as returned by getwavegrps, scale the peak
	amplitude of each waveform wavegrps[t,r][i] by:

	* If individual is False, the maximum peak amplitude of all waveforms
	  in wavegrps[t,r], or

	* If individual is True, by the waveform's own peak amplitude.

	If the waveform peak amplitude is less than the maximum peak amplitude
	in all wavegrps time sqrt(sys.float_info.epsilon), that waveform will
	not be scaled.

	The equalization is done in place, but wavegrps is also returned.
	'''
	# Find the peak amplitudes for each group
	pkamps = { k: [wf.envelope().extremum()[0] for wf in v]
			for k, v in wavegrps.items() }

	minamp = sqrt(sys.float_info.epsilon)

	if not individual:
		# Reduce all peak amplitudes to one per group
		pkamps = { k: max(v) for k, v in pkamps.items() }
		
		# Find low-amplitude threshold
		minamp *= max(pkamps.values())

		# Equalize the waveforms in each group, if desired
		for k, pamp in pkamps.items():
			if pamp < minamp: continue
			for v in wavegrps[k]: v /= pamp
	else:
		# Find the low-amplitude threshold
		minamp *= max(max(v) for v in pkamps.values())

		for k, pamp in pkamps.items():
			for i, v in enumerate(pamp):
				if v < minamp: continue
				wavegrps[k][i] /= v

	return wavegrps


def getwavegrps(infiles, nsamp=None):
	'''
	For a sequence infiles of input WaveformSet files that each contain a
	single waveform, prepare a mapping from transmit-receiver pairs to a
	list of Waveform objects representing backscatter waves observed at the
	pair. The pair is given by (wset.rxidx[0], wset.txidx.next()) for each
	file. Waveforms are ordered according to a lexicographical ordering of
	infiles. If nsamp is not None, the nsamp property of each Waveform
	object will be overridden.

	Only element indices whose Waveform lists have a length that matches
	that of the longest Waveform list will be included.
	'''
	wavegrps = defaultdict(list)

	for infile in sorted(infiles):
		wset = WaveformSet.fromfile(infile, force_dtype='float64')
		f2c = wset.f2c

		if wset.ntx != 1 or wset.nrx != 1:
			raise IOError(f'Input {infile} must contain a single waveform')

		(tx, rx), wf = next(wset.allwaveforms())
		if nsamp: wf.nsamp = nsamp

		# Remove F2C
		dwin = wf.datawin
		nwf = Waveform(wf.nsamp + f2c, wf.getsignal(dwin), dwin.start + f2c)
		wavegrps[tx,rx].append(nwf)

	# Filter the list to exclude short lists
	maxlen = max(len(w) for w in wavegrps.values())
	return { k: v for k, v in wavegrps.items() if len(v) == maxlen }


def getwave(infile, nsamp=None):
	'''
	For an input WaveformSet file infile that contains a single waveform,
	return the (t, r) index pair stored in the file and the waveform it
	contains. The (t, r) index pair is given by the tuple
	(wset.rxidx[0], wset.txidx.next()).

	If nsamp is not None, the nsamp property of each read wavefrom is
	replaced with the specified value.

	If equalize is True, each waveform will be scaled by the inverse of its
	peak amplitude as long as the peak amplitude is larger than the value
	sqrt(sys.float_info.epsilon).

	An IOError will be raised if the file contains more than one waveform.
	'''
	wset = WaveformSet.fromfile(infile, force_dtype='float64')
	f2c = wset.f2c

	if wset.ntx != 1 or wset.nrx != 1:
		raise IOError(f'Input {infile} must contain a single waveform')

	(tx, rx), wf = next(wset.allwaveforms())
	if nsamp: wf.nsamp = nsamp

	# Shift out the F2C
	dwin = wf.datawin
	wf = Waveform(wf.nsamp + f2c, wf.getsignal(dwin), dwin.start + f2c)

	return (tx, rx), wf


if __name__ == '__main__':
	parser = ArgumentParser(description='Plot waveforms in videos or PDF images')

	parser.add_argument('-l', '--log', action='store_true',
			help='Display log-magnitude instead of linear amplitude')

	parser.add_argument('-z', '--zero', action='store_true',
			help='Zero waveforms with no arrival times')

	parser.add_argument('-s', '--suppress', action='store_true',
			help='Eliminate waveforms with no arrival times')

	parser.add_argument('-w', '--window', nargs=2, type=int,
			default=None, metavar=('START', 'END'),
			help='Only display samples from START to END '
				'(relative to arrival times if provided)')

	parser.add_argument('-a', '--atimes', nargs='+', default=None,
			help='Arrival-time files to align waveforms')

	parser.add_argument('-c', '--cols', nargs='+', type=int, default=None,
			help='Columns of arrival-time records to use for alignment')

	parser.add_argument('-t', '--thresh', type=float, default=None,
			help='Color (image) or y-axis (video) threshold')

	parser.add_argument('-f', '--freq', type=float, default=20.,
			help='Frequency of samples in waveform files')

	parser.add_argument('-n', '--nsamp', type=int, default=None,
			help='Force all waveform files to have NSAMP samples')

	parser.add_argument('-b', '--bitrate', type=int, default=-1,
			help='Set bitrate for video output in kbps')

	parser.add_argument('--one-sided', action='store_true',
			help='Use a one-sided color or amplitude scale')

	parser.add_argument('-e', '--equalize', action='count',
			help='Equalize waveforms (in videos, use twice to '
				'equalize all waves in each frame independently')

	parser.add_argument('output', type=str,
			help='Name of output file (PDF for image, mp4 for video)')

	parser.add_argument('inputs', type=str, nargs='+',
			help='Names of waveform input files')

	args = parser.parse_args(sys.argv[1:])

	# Determine the output mode
	imgext = os.path.splitext(args.output)[1].lower()
	if imgext == '.mp4': vidmode = True
	elif imgext == '.pdf': vidmode = False
	else: sys.exit(f'ERROR: Output {args.output} is not an MP4 or PDF')

	if vidmode:
		if args.log:
			sys.exit('ERROR: Cannot set --log for video output')
		elif args.zero:
			sys.exit('ERROR: Cannot set --zero for video output')

	try: args.inputs = matchfiles(args.inputs)
	except IOError as e: sys.exit(f'ERROR: {e}')

	if args.atimes:
		# Load arrival times and convert to samples
		args.atimes = getatimes(args.atimes, args.freq, not vidmode, args.cols)
		print(f'Parsed {len(args.atimes)} arrival times')
	elif args.suppress or args.zero:
		sys.exit('ERROR: Cannot set --suppress or --zero without --atimes')

	if vidmode:
		# Load the backscatter waves in groups by element
		wavegrps = getwavegrps(args.inputs, args.nsamp)
		# Shift waveforms if arrival times are provided
		if args.atimes:
			wavegrps = shiftgrps(wavegrps, args.atimes, args.suppress)
			print('Shifted waveform groups')
		print('Storing waveform video to file', args.output)
		plotframes(args.output, wavegrps, args.atimes,
				args.window, args.equalize,
				args.thresh, args.bitrate, args.one_sided)
	else:
		# Load the waveforms
		waves = dict(getwave(inf, args.nsamp) for inf in args.inputs)

		# There is no mean arrival time unless arrival times are provided
		mtime = None

		if args.atimes:
			# Find the mean arrival time for all waveforms
			celts = set(waves).intersection(args.atimes)
			print(f'{len(celts)} waveforms have associated arrival times')
			mtime = int(np.mean([args.atimes[c] for c in celts]))

			if args.suppress: print('Will suppress unaligned waveforms')
			elif args.zero: print('Will zero unaligned waveforms')

			# Define the relative window in alignment mode
			if args.window is not None:
				start, end = args.window
				start = max(0, mtime + start)
				end = mtime + end
				args.window = Window(start, end=end)
		elif args.window is not None:
			# Define the absolute window
			start, end = args.window
			args.window = Window(start, end=end, nonneg=True)

		# Align, window and equalize each waveform as necessary
		pwaves = { }
		for k, wave in waves.items():
			# Try to shift waveforms to mean arrival time
			try:
				atime = args.atimes[k]
			except (KeyError, TypeError):
				if args.suppress:
					continue
				elif args.zero:
					pwaves[k] = Waveform(wave.nsamp)
					continue
			else:
				wave = wave.shift(mtime - atime)

			if args.window is not None:
				wave = wave.window(args.window)
			if args.equalize:
				pkamp = wave.envelope().extremum()[0]
				if pkamp > sqrt(sys.float_info.epsilon): wave /= pkamp

			# Store the final product
			pwaves[k] = wave

		waves = pwaves
		print('Processed waveforms, storing to file', args.output)

		plotwaves(args.output, waves, args.atimes, mtime,
				args.window, args.log, args.thresh, args.one_sided)
