#!/usr/bin/env python 

import numpy as np, getopt, sys, os

from scipy.signal import hilbert

from itertools import izip

from collections import defaultdict

from habis.habiconf import matchfiles
from habis.sigtools import Waveform, Window
from habis.formats import loadkeymat, WaveformSet

def usage(progname=None, fatal=False):
	if progname is None: progname = sys.argv[0]
	print >> sys.stderr, 'USAGE: %s [-b bitrate] [-m] [-w s,e] [-a file[:column]] [-t thresh] [-f freq] [-n nsamp] <imgname> <wavesets>' % progname
	sys.exit(fatal)


def plotframes(output, waves, atimes, dwin=None, cthresh=None, bitrate=-1):
	'''
	Prepare, using the ffmpeg writer in matplotlib.animation, a video in
	which each frame depicts aligned waveforms received by a common
	element. The argument waves should be a mapping from element index to a
	list of Waveform objects. The argument atimes should be a mapping from
	element index to a list of arrival times (one for each Waveform in the
	list of Waveforms in an entry of waves). For each element, all
	waveforms will be aligned to the first one using the times in atimes.
	(If atimes is not specified or does not contain times for a given
	element, no alignment will be attempted.)
	'''
	import matplotlib as mpl
	mpl.use('agg')
	import matplotlib.pyplot as plt
	import matplotlib.animation as ani

	# Prepare the axes for a 1080p frame
	fig = plt.figure()
	fig.set_dpi(80)
	fdpi = float(fig.get_dpi())
	fig.set_size_inches(1920. / fdpi, 1080. / fdpi)

	# Grab the axes
	ax = ax = fig.add_subplot(111)

	# Prepare the video writer
	try: ffwr = ani.writers['ffmpeg']
	except KeyError:
		raise KeyError('The ffmpeg animation writer is required for video creation')

	# Configure the writer (let ffmpeg decide the bitrate)
	metadata = dict(title='Waveform analysis video', artist='waveviz.py')
	writer = ffwr(fps=5, bitrate=bitrate, metadata=metadata)

	if dwin is None or atimes is None:
		# With no data window, show the entire data range
		dstart = min(w.datawin.start for v in waves.itervalues() for w in v)
		dend = max(w.datawin.end for v in waves.itervalues() for w in v)
		dwin = Window(dstart, end=dend)
	else:
		# With a data window, encompass the maximum range of arrival times
		dstart = int(min(atimes[elt][0] if elt in atimes else float('inf')
				for elt in waves.iterkeys()))
		dend = int(max(atimes[elt][0] if elt in atimes else 0
				for elt in waves.iterkeys()))
		dwin = Window(max(0, dstart + dwin[0]), end=dend + dwin[1])

	# Build the common time axis
	taxis = np.arange(dwin.start, dwin.end)

	# Clip the waveforms to the common data window
	waves = { k: [ w.window(dwin) for w in v ] for k, v in waves.iteritems() }

	# Set the amplitude limits
	pkamps = [ w.envelope().extremum()[0] for v in waves.itervalues() for w in v ]
	if cthresh is None: vmax = np.max(pkamps)
	else: vmax = np.mean(pkamps) + cthresh * np.std(pkamps)

	# Ensure all data sets are equally sized
	nsets = max(len(v) for v in waves.itervalues())
	if any(len(v) != nsets for v in waves.itervalues()):
		raise ValueError('All waveform lists must be equally sized')

	# Create the frames and write the video
	with writer.saving(fig, output, fig.get_dpi()):
		# Create the empty plot for efficiency
		lines = ax.plot(*[[] for i in range(2 * nsets)])
		ax.axis([dwin.start, dwin.end, -vmax, vmax])
		ax.set_xlabel('Time, samples', fontsize=14)
		ax.set_ylabel('Amplitude', fontsize=14)
		ax.grid(True)

		for i, (elt, wlist) in enumerate(sorted(waves.iteritems())):
			# Update the line data
			for l, w in izip(lines, wlist):
				l.set_data(taxis, w.getsignal(dwin))
			ax.set_title('Waveform for element %d' % elt, fontsize=16)

			# Capture the frame
			writer.grab_frame()
			if not i % 50: print 'Stored frame %d' % elt


def plotwaves(output, waves, atimes=None, dwin=None, cthresh=None):
	'''
	Plot, into the image file output, the habis.sigtools.Waveform objects
	mapped (by index) in waves, with temporal variations along the vertical
	axis. The index along the horizontal axis is into sorted(waves) and is
	not guaranteed to correspond to the index in the mapping.

	If atimes is not None, it should map indices to waveform arrival times.
	A subplot will show these arrival times on the same horizontal axis as
	the waveform image. Elements in waves that do not exist in atimes will
	be replaced with NaN when plotting arrival times.

	The waves are cropped to the specified data window prior to plotting.
	If dwin is None, the smallest data window that encompasses all plotted
	signals will be used.

	The color range to clip at cthresh standard deviations above the mean
	peak amplitude over all signals. If cthresh is None, the narrowest
	range that avoids clipping will be selected.
	'''
	import matplotlib as mpl
	mpl.use('pdf')
	import matplotlib.pyplot as plt
	from matplotlib import cm

	# Split the mapping in indexed order
	widx, waves = zip(*sorted(waves.iteritems()))

	# Pull the relevant arrival times for a subplot
	if atimes is not None:
		atimes = [atimes[k] if k in atimes else float('nan') for k in widx]

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
	pkamps = np.max(np.abs(hilbert(img, axis=1)), axis=1)

	if cthresh is None: cmax = np.max(pkamps)
	else: cmax = np.mean(pkamps) + cthresh * np.std(pkamps)

	# Shift extent by half a pixel so grid lines are centered on samples
	extent = [-0.5, img.shape[0] - 0.5, dwin[0] - 0.5, dwin[0] + dwin[1] - 0.5]

	# Plot the waveform image
	ax[0].imshow(img.T, vmin=-cmax, vmax=cmax, cmap=cm.bone,
			interpolation='nearest', origin='lower', extent=extent)
	ax[0].grid(True)
	ax[0].set_aspect('auto')
	ax[0].set_ylabel('Sample index', fontsize=16)
	ax[0].set_title('Waveforms aligned to mean arrival time', fontsize=16)

	if atimes is not None:
		# Plot the arrival-time image
		ax[1].plot(atimes, linewidth=0.5)
		ax[1].grid(True)
		ax[1].set_xlabel('Waveform index', fontsize=16)
		ax[1].set_ylabel('Arrival time, samples', fontsize=16)
		ax[1].set_title('Waveform arrival times', fontsize=16)
		plt.setp(ax[0].get_xticklabels(), visible=False)

	# Ensure at least 10 x ticks exist
	ax[0].set_xlim(0, img.shape[0])
	if len(ax[0].get_xticks()) < 10:
		ax[0].set_xticks(range(0, img.shape[0] + 1, int(img.shape[0] / 10)))

	# Save the image
	fig.savefig(output, bbox_inches='tight')


def getatimes(atarg, freq=1, scalar=True):
	'''
	Given a file name with an optional ":<column>,<column>,..." suffix of
	integer indices, try to open the arrival-time map with
	habis.formats.loadkeymat() and pull the specified column.

	A file with the full name in atarg will first be checked. If none
	exists or the file cannot be opened, and the name constains a suffix of
	integer indices, a file whose name matches atarg up to the suffix will
	be checked and the specified column will be pulled.

	If no suffix is unspecified, the first column will be pulled.

	The times are scaled by the frequency to convert the times to samples.

	If scalar is True, values in the returned map will be scalars.
	Otherwise, the returned values will be arrays.
	'''
	acols = [0]
	try:
		# Try to load the file with the full name
		atmap = loadkeymat(atarg, scalar=False)
	except IOError as err:
		atcomps = atarg.split(':')
		if len(atcomps) < 2: raise err
		# Treat the name as a name + column
		acols = [int(av, base=10) for av in atcomps[-1].split(',')]
		atname = ':'.join(atcomps[:-1])
		atmap = loadkeymat(atname, scalar=False)
		print 'Loading columns %s from arrival-time file %s' % (acols, atname)

	if scalar:
		if len(acols) != 1:
			raise ValueError('Scalar arrival-time map requires a single column specification')
		acols = acols[0]

	return { k: freq * v[acols] for k, v in atmap.iteritems() }


def getbsgroups(infiles, atimes=None, nsamp=None):
	'''
	For a sequence infiles of input WaveformSet files, prepare a mapping
	from element indices to a list of Waveform objects representing
	backscatter waves observed at the indexed element. The Waveforms are
	ordered according to a lexicographical ordering of infiles. If nsamp is
	not None, the nsamp property of each Waveform object will be
	overridden.

	If atimes is not None, it should map element indices to a list of
	arrival times (one for each unique Waveform in the set of Waveforms
	received by the element). If an entry exists in atimes for a given
	element, all waveforms for that element will be aligned to the first
	Waveform for the element (using only the arrival times).

	Only element indices whose Waveform lists have a length that matches
	that of the longest Waveform list will be included.
	'''
	bswaves = defaultdict(list)

	for infile in sorted(infiles):
		wset = WaveformSet.fromfile(infile)
		f2c = wset.f2c

		for rx in wset.rxidx:
			wf = wset.getwaveform(rx, rx, maptids=True)
			if nsamp: wf.nsamp = nsamp

			try: atlist = atimes[rx]
			except (KeyError, TypeError): pass
			else: wf = wf.shift(atlist[0] - atlist[len(bswaves[rx])])

			dwin = wf.datawin
			nwf = Waveform(wf.nsamp + f2c, wf.getsignal(dwin), dwin.start + f2c)
			bswaves[rx].append(nwf)

	maxlen = max(len(w) for w in bswaves.itervalues())
	return { k: v for k, v in bswaves.iteritems() if len(v) == maxlen }


def getbswaves(infile, atimes=None, nsamp=None):
	'''
	For an input WaveformSet file infile, return a dictionary mapping
	receive-channel indices to backscatter Waveforms. All data windows are
	adjusted to 0 f2c.

	If atimes is not None, it should map element indices to arrival times.
	In this case, every waveform will be aligned to the average arrival
	time if the waveform has an entry in atimes. (If a waveform is not
	listed in atimes, it will not be shifted.)

	If nsamp is not None, the nsamp property of each read wavefrom is
	replaced with the specified value.
	'''
	wset = WaveformSet.fromfile(infile)
	f2c = wset.f2c

	if atimes: mtime = np.mean(atimes.values())

	bswaves = { }
	for rx in wset.rxidx:
		wf = wset.getwaveform(rx, rx, maptids=True)
		if nsamp: wf.nsamp = nsamp

		try: shift = mtime - atimes[rx]
		except (KeyError, TypeError): pass
		else: wf = wf.shift(shift)

		dwin = wf.datawin

		bswaves[rx] = Waveform(wf.nsamp + f2c, wf.getsignal(dwin), dwin.start + f2c)

	return bswaves


if __name__ == '__main__':
	dwin = None
	nsamp = None
	cthresh = None
	zeropad = False
	atimes = None
	freq = 20.
	hidewf = False
	bitrate = -1

	optlist, args = getopt.getopt(sys.argv[1:], 'hw:a:t:f:n:p:b:m')

	for opt in optlist:
		if opt[0] == '-h':
			usage(fatal=False)
		elif opt[0] == '-w':
			dwin = [int(s, base=10) for s in opt[1].split(',')]
			if len(dwin) != 2:
				raise ValueError('Window must be a start,end pair')
		elif opt[0] == '-a':
			atimes = opt[1]
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
		else:
			usage(fatal=True)

	if len(args) < 2:
		print >> sys.stderr, 'ERROR: required arguments missing'
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
		print >> sys.stderr, 'ERROR: <imgname> must have extension mp4 or pdf'
		usage(fatal=True)

	if atimes is not None:
		# Load arrival times and convert to samples
		atimes = getatimes(atimes, freq, not vidmode)

	try:
		infiles = matchfiles(args)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(fatal=True)

	if vidmode:
		# Load the backscatter waves in groups by element
		bswaves = getbsgroups(infiles, atimes, nsamp)
		print 'Storing waveform video to file', imgname
		plotframes(imgname, bswaves, atimes, dwin, cthresh, bitrate)
	else:
		# Load and align all waves
		bswaves = dict(kp for infile in infiles
				for kp in getbswaves(infile, atimes, nsamp).iteritems())

		if not atimes:
			if dwin is not None:
				# Convert the window specification to a Window object
				dwin = Window(dwin[0], end=dwin[1], nonneg=True)
		else:
			if dwin is not None:
				# The window specification is relative to the mean time
				mtime = int(np.mean(atimes.values()))
				dwin = Window(max(0, mtime + dwin[0]), end=mtime + dwin[1])

		if hidewf:
			print 'Suppressing unaligned waveforms'
			for k, v in bswaves.iteritems():
				if k not in atimes: v *= 0

		print 'Storing waveform image to file', imgname
		plotwaves(imgname, bswaves, atimes, dwin, cthresh)
