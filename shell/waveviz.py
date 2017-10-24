#!/usr/bin/env python 

import numpy as np, getopt, sys, os

from math import sqrt

from scipy.signal import hilbert



from collections import defaultdict

from habis.habiconf import matchfiles
from habis.sigtools import Waveform, Window
from habis.formats import loadmatlist, WaveformSet

def usage(progname=None, fatal=False):
	if progname is None: progname = sys.argv[0]
	print('USAGE: %s [-b bitrate] [-e] [-m] [-l] [-w s,e] [-a glob[:column]] [-t thresh] [-f freq] [-n nsamp] <imgname> <wavesets>' % progname, file=sys.stderr)
	sys.exit(fatal)


def plotframes(output, waves, atimes, dwin=None, cthresh=None, bitrate=-1):
	'''
	Prepare, using the ffmpeg writer in matplotlib.animation, a video in
	which each frame depicts aligned waveforms received by a common
	element. The argument waves should be a mapping from transmit-receive
	pairs to a list of Waveform objects, prealigned if alignment is
	desired.

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
	'''
	import matplotlib as mpl
	mpl.use('agg')
	import matplotlib.pyplot as plt
	import matplotlib.animation as ani

	# Ensure all data sets are equally sized
	nsets = max(len(v) for v in waves.values())
	if any(len(v) != nsets for v in waves.values()):
		raise ValueError('All waveform lists must be equally sized')

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
			dstart = min(atimes[pair][0] for pair in cpairs)
			dend = max(atimes[pair][0] for pair in cpairs)
			dwin = Window(max(0, int(dstart + dwin[0])), end=int(dend + dwin[1]))
		else:
			dwin = Window(dwin[0], end=dwin[1])

	# Clip the waveforms to the common data window
	waves = { k: [ w.window(dwin) for w in v ] for k, v in waves.items() }

	# Set the amplitude limits
	pkamps = [ w.envelope().extremum()[0] for v in waves.values() for w in v ]
	if cthresh is None: vmax = np.max(pkamps)
	else: vmax = np.mean(pkamps) + cthresh * np.std(pkamps)

	# Build the common time axis
	taxis = np.arange(dwin.start, dwin.end)

	print('Display frame is [%d, %d, %g, %g]' % (dwin.start, dwin.end, -vmax, vmax))

	# Create the frames and write the video
	with writer.saving(fig, output, fig.get_dpi()):
		# Create the empty plot for efficiency
		lines = ax.plot(*[[] for i in range(2 * nsets)])
		apoint, = ax.plot([], [], 'bs')
		ax.axis([taxis[0], taxis[-1], -vmax, vmax])
		ax.set_xlabel('Time, samples', fontsize=14)
		ax.set_ylabel('Amplitude', fontsize=14)
		ax.grid(True)

		for i, (pair, wlist) in enumerate(sorted(waves.items())):
			# Update the line data
			for l, w in zip(lines, wlist):
				l.set_data(taxis, w.getsignal(dwin))

			# Plot an arrival time, if possible
			try:
				atelt = int(atimes[pair][0])
			except (KeyError, TypeError):
				apoint.set_visible(False)
			else:
				apoint.set_data([atelt], [wlist[0][atelt]])
				apoint.set_visible(True)

			ax.set_title('Waveform %s' % (pair,), fontsize=14)

			# Capture the frame
			writer.grab_frame()
			if not i % 50: print('Stored frame %s' % (pair,))


def plotwaves(output, waves, atimes=None, mtime=None,
		dwin=None, log=False, cthresh=None):
	'''
	Plot, into the image file output, the habis.sigtools.Waveform objects
	mapped (by index) in waves, with temporal variations along the vertical
	axis. The index along the horizontal axis is into sorted(waves) and is
	not guaranteed to correspond to the index in the mapping.

	If atimes is not None, it should map indices to waveform arrival times.
	A subplot will show these arrival times on the same horizontal axis as
	the waveform image. Elements in waves that do not exist in atimes will
	be replaced with NaN when plotting arrival times.

	If mtimes is not None, it should be the mean arrival time use to align
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
	'''
	import matplotlib as mpl
	mpl.use('pdf')
	import matplotlib.pyplot as plt
	from matplotlib import cm

	# Split the mapping in indexed order
	widx, waves = zip(*sorted(waves.items()))

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

		clim = [-cmax, cmax]
		cmap = cm.RdBu
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
		if mtime: title += ' (%s samples)' % (mtime,)
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
		ax[0].set_xticks(list(range(0, img.shape[0] + 1, int(img.shape[0] / 10))))

	# Save the image
	fig.savefig(output, bbox_inches='tight')


def getatimes(atarg, freq=1, scalar=True):
	'''
	Given a glob with an optional ":<column>,<column>,..." suffix of
	integer indices, try to open arrival-time maps matching the glob with
	habis.formats.loadmatlist() and pull the specified columns. Keys of the
	map should be transmit-receive pairs (t, r).

	A glob matching the full name in atarg will first be checked. If no
	matches exist or the file cannot be opened, and the name
	constains a suffix of
	integer indices, a file whose name matches atarg up to the suffix will
	be checked and the specified column will be pulled.

	If no suffix is specified, the first column will be pulled.

	The times are scaled by the frequency to convert the times to samples.

	If scalar is True, values in the returned map will be scalars.
	Otherwise, the returned values will be arrays.
	'''
	acols = [0]
	try:
		# Try to load the file with the full name
		atmap = loadmatlist(atarg, scalar=False, nkeys=2, forcematch=True)
	except IOError as err:
		atcomps = atarg.split(':')
		if len(atcomps) < 2: raise err
		# Treat the name as a name + column
		acols = [int(av, base=10) for av in atcomps[-1].split(',')]
		atname = ':'.join(atcomps[:-1])
		atmap = loadmatlist(atname, scalar=False, nkeys=2, forcematch=True)
		print('Loading columns %s from arrival-time file %s' % (acols, atname))

	if scalar:
		if len(acols) != 1:
			raise ValueError('Scalar arrival-time map requires a single column specification')
		acols = acols[0]

	return { k: freq * v[acols] for k, v in atmap.items() }


def shiftgrps(wavegrps, atimes):
	'''
	In a mapping wavegrps as returned by getwavegrps, shift each waveform
	wavegrps[t,r][i] by the difference atimes[t,r][0] - atimes[t,r][i]. If
	a list atimes[t,r] cannot be found, no shift will be performed for that
	(t, r) pair.

	If atimes[t,r] exists but its length does not match the length of
	wavegrps[t,r], an IndexError will be raised.

	The shifting is done in place, but wavegrps is also returned.
	'''
	for (t,r) in wavegrps.keys():
		# Pull the waveform list
		waves = wavegrps[t,r]

		# Pull the arrival-time list, if possible
		try: atlist = atimes[t,r]
		except KeyError: continue

		if len(atlist) != len(waves):
			raise IndexError('Length of arrival-time list does not match length of wave-group list for pair %s' % ((t,r),))

		# Build the new list of shifted waves
		wavegrps[t,r] = [ waves[0] ]
		wavegrps[t,r].extend(wf.shift(atlist[0] - atv)
					for wv, atv in zip(waves[1:], atlist[1:]))

	return wavegrps


def eqwavegrps(wavegrps):
	'''
	In a mapping wavegrps as returned by getwavegrps, scale the peak
	amplitude of each waveform wavegrps[t,r][i] by the inverse of the
	maximum peak amplitude of all waveforms in wavegrps[t,r], unless the
	maximum value is less than sqrt(sys.float_info.epsilon) times the
	maximum amplitude across all waveforms in wavegrps.

	The shifting is done in place, but wavegrps is also returned.
	'''
	# Find the peak amplitudes for each group
	pkamps = { k: max(wf.envelope().extremum()[0] for wf in v) 
			for k, v in wavegrps.items() }

	# Find low-amplitude threshold
	minamp = max(pkamps.values()) * sqrt(sys.float_info.epsilon)

	# Equalize the waveforms in each group, if desired
	for k, pamp in pkamps.items():
		if pamp < minamp: continue
		for v in wavegrps[k]: v /= pamp

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
		wset = WaveformSet.fromfile(infile)
		f2c = wset.f2c

		if wset.ntx != 1 or wset.nrx != 1:
			raise IOError('Input %s must contain a single waveform' % (infile,))

		rx = wset.rxidx[0]
		tx = next(wset.txidx)

		wf = wset.getwaveform(rx, tx)
		if nsamp: wf.nsamp = nsamp

		# Remove F2C
		dwin = wf.datawin
		nwf = Waveform(wf.nsamp + f2c, wf.getsignal(dwin), dwin.start + f2c)
		wavegrps[tx,rx].append(nwf)

	# Filter the list to exclude short lists
	maxlen = max(len(w) for w in wavegrps.values())
	return { k: v for k, v in wavegrps.items() if len(v) == maxlen }


def getwave(infile, nsamp=None, equalize=False):
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
	wset = WaveformSet.fromfile(infile)
	f2c = wset.f2c

	if wset.ntx != 1 or wset.nrx != 1:
		raise IOError('Input %s must contain a single waveform' % (infile,))

	rx = wset.rxidx[0]
	tx = next(wset.txidx)

	wf = wset.getwaveform(rx, tx)
	if nsamp: wf.nsamp = nsamp

	# Shift out the F2C
	dwin = wf.datawin 
	wf = Waveform(wf.nsamp + f2c, wf.getsignal(dwin), dwin.start + f2c)

	if equalize:
		pkamp = wf.envelope().extremum()[0]
		if pkamp > sqrt(sys.float_info.epsilon): wf /= pkamp

	return (tx, rx), wf


if __name__ == '__main__':
	dwin = None
	nsamp = None
	cthresh = None
	log = False
	zeropad = False
	atimes = None
	freq = 20.
	hidewf = False
	bitrate = -1
	equalize = False

	optlist, args = getopt.getopt(sys.argv[1:], 'hw:a:t:f:n:p:b:mle')

	for opt in optlist:
		if opt[0] == '-h':
			usage(fatal=False)
		elif opt[0] == '-w':
			dwin = [int(s, base=10) for s in opt[1].split(',')]
			if len(dwin) != 2:
				raise ValueError('Window must be a start,end pair')
		elif opt[0] == '-a':
			atimes = opt[1]
		elif opt[0] == '-l':
			log = True
		elif opt[0] == '-t':
			cthresh = float(opt[1])
		elif opt[0] == '-f':
			freq = float(opt[1])
		elif opt[0] == '-n':
			nsamp = int(opt[1])
		elif opt[0] == '-b':
			bitrate = int(opt[1])
		elif opt[0] == '-m':
			hidewf = True
		elif opt[0] == '-e':
			equalize = True
		else:
			usage(fatal=True)

	if len(args) < 2:
		print('ERROR: required arguments missing', file=sys.stderr)
		usage(fatal=True)

	# Pull the output name and load the input files
	imgname = args.pop(0)

	# Determine the mode
	imgext = os.path.splitext(imgname)[1].lower()
	if imgext == '.mp4':
		vidmode = True
	elif imgext == '.pdf':
		vidmode = False
	else:
		print('ERROR: <imgname> must have extension mp4 or pdf', file=sys.stderr)
		usage(fatal=True)

	if atimes is not None:
		# Load arrival times and convert to samples
		atimes = getatimes(atimes, freq, not vidmode)
		print('Parsed arrival times')

	try:
		infiles = matchfiles(args)
	except IOError as e:
		print('ERROR:', e, file=sys.stderr)
		usage(fatal=True)

	if vidmode:
		# Load the backscatter waves in groups by element
		wavegrps = getwavegrps(infiles, nsamp)
		# Shift waveforms if arrival times are provided
		if atimes:
			wavegrps = shiftgrps(wavegrps, atimes)
			print('Shifted waveform groups')
		# Equalize waveforms as desired
		if equalize:
			wavegrps = eqwavegrps(wavegrps)
			print('Equalized waveform groups')
		print('Storing waveform video to file', imgname)
		plotframes(imgname, wavegrps, atimes, dwin, cthresh, bitrate)
	else:
		# Load the waveforms
		waves = dict(getwave(inf, nsamp, equalize) for inf in infiles)

		# There is no mean arrival time unless arrival times are provided
		mtime = None

		if atimes:
			# Find the mean arrival time for all waveforms
			celts = set(waves).intersection(atimes)
			mtime = int(np.mean([atimes[c] for c in celts]))
			# Shift all waveforms with an arrival time to the mean
			for k in celts:
				waves[k] = waves[k].shift(mtime - atimes[k])
			print('Shifted waveforms')
			if dwin is not None:
				# The window specification is relative to the mean time
				dwin = Window(max(0, mtime + dwin[0]), end=mtime + dwin[1])
		elif dwin is not None:
			# Convert the window specification to a Window object
			dwin = Window(dwin[0], end=dwin[1], nonneg=True)

		if hidewf and atimes:
			print('Suppressing unaligned waveforms')
			for k, v in waves.items():
				if k not in atimes: v *= 0

		print('Storing waveform image to file', imgname)
		plotwaves(imgname, waves, atimes, mtime, dwin, log, cthresh)
