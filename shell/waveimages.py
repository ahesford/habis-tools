#!/usr/bin/env python 

import numpy as np, getopt, sys
import multiprocessing

from scipy.signal import hilbert

from itertools import izip

from habis.habiconf import matchfiles
from habis.sigtools import Waveform, Window
from habis.formats import loadkeymat, WaveformSet

def usage(progname=None, fatal=False):
	if progname is None: progname = sys.argv[0]
	print >> sys.stderr, 'USAGE: %s [-p nproc] [-m] [-w s,e] [-a file[:column]] [-t thresh] [-f freq] [-n nsamp] [-b start:end[:tails]] <imgname> <wavesets>' % progname
	sys.exit(fatal)


def plotwaves(output, waves, dwin=None, cthresh=None):
	'''
	Plot, into the image file output, the habis.sigtools.Waveform objects
	in the sequence waves with temporal variations along the vertical axis
	and waveform indices along the horizontal axis.

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

	if dwin is None:
		dstart = min(w.datawin.start for w[0] in waves)
		dend = max(w.datawin.end for w[0] in waves)
		dwin = Window(dstart, end=dend)

	# Prepare the axes
	fig = plt.figure()

	# Size the figure so each sample gets at least 1 pixel
	fig.set_dpi(220)
	fdpi = float(fig.get_dpi())
	spwfrac = fig.subplotpars.right - fig.subplotpars.left
	sphfrac = (fig.subplotpars.top - fig.subplotpars.bottom - fig.subplotpars.hspace) / 2.
	wfig = max(12., np.ceil(float(len(waves)) / spwfrac / fdpi))
	hfig = max(np.ceil(float(dwin[1]) / sphfrac / fdpi), wfig / 3.)
	fig.set_size_inches(wfig, hfig)

	# Add the figure
	ax = [fig.add_subplot(211)]
	ax.append(fig.add_subplot(212, sharex=ax[0]))

	# Pull the waveforms and determine the color range
	img = np.array([w[0].getsignal(dwin) for w in waves])
	pkamps = np.max(np.abs(hilbert(img, axis=1)), axis=1)

	if cthresh is None: cmax = np.max(pkamps)
	else: cmax = np.mean(pkamps) + cthresh * np.std(pkamps)

	# Shift extent by half a pixel so grid lines are centered on samples
	extent = [-0.5, img.shape[0] - 0.5, dwin[0] - 0.5, dwin[0] + dwin[1] - 0.5]

	# Plot the waveform image
	ax[0].imshow(img.T, vmin=-cmax, vmax=cmax, cmap=cm.bone,
			interpolation='nearest', origin='lower', extent=extent)

	plt.setp(ax[0].get_xticklabels(), visible=False)
	ax[0].grid(True)
	ax[0].set_aspect('auto')
	ax[0].set_ylabel('Sample index', fontsize=16)
	ax[0].set_title('Waveforms aligned to mean arrival time', fontsize=16)

	# Plot the arrival-time image
	ax[1].plot([w[1] for w in waves], linewidth=0.5)
	ax[1].grid(True)

	ax[1].set_xlabel('Waveform index', fontsize=16)
	ax[1].set_ylabel('Arrival time, samples', fontsize=16)
	ax[1].set_title('Waveform arrival times', fontsize=16)

	# Ensure at least 10 x ticks exist
	ax[1].set_xlim(0, img.shape[0])
	if len(ax[1].get_xticks()) < 10:
		ax[1].set_xticks(range(0, img.shape[0] + 1, int(img.shape[0] / 10)))

	# Save the image
	fig.savefig(output, bbox_inches='tight')


def getatimes(atarg, freq=1):
	'''
	Given a file name with an optional ":<column>" integer suffix, try to
	open the arrival-time map with habis.formats.loadkeymat() and pull the
	specified column.

	A file with the full name in atarg will first be checked. If none
	exists or the file cannot be opened, and the name constains a
	":<column>" suffix where <column> is an integer, a file whose name
	matches atarg up to the suffix will be checked and the specified column
	will be pulled.

	If column is unspecified, the first column is used.

	The times are scaled by the frequency to convert the times to samples.
	'''
	acol = 0
	try:
		# Try to load the file with the full name
		atmap = loadkeymat(atarg, scalar=False)
	except IOError as err:
		atcomps = atarg.split(':')
		if len(atcomps) < 2: raise err
		# Treat the name as a name + column
		acol = int(atcomps[-1])
		atmap = loadkeymat(':'.join(atcomps[:-1]), scalar=False)

	return { k: freq * v[acol] for k, v in atmap.iteritems() }


def getbswaves(args):
	'''
	For args = (infile, atimes=None, bpass=None, nsamp=None), where infile
	specifies a WaveformSet file, return a dictionary mapping
	receive-channel indices to backscatter Waveforms. All data windows are
	adjusted to 0 f2c.

	If atimes is not None, it should map element indices to arrival times.
	In this case, every waveform will be aligned to the average arrival
	time if the waveform has an entry in atimes. (If a waveform is not
	listed in atimes, it will not be shifted.)
	
	If bpass is not None, it should be a tuple (start, end, [tails]) which
	will be passed to Waveform.bandpass() upon reading.

	If nsamp is not None, and bpass is not None, the nsamp property of each
	read wavefrom is replaced with the specified value. The value is
	ignored when bpass is None.
	'''
	if len(args) > 4: raise TypeError('Unrecognized argument')

	infile = args[0]

	try: atimes = args[1]
	except IndexError: atimes=None

	try: bpass = args[2]
	except IndexError: bpass = None

	try: nsamp = args[3]
	except IndexError: nsamp = None

	wset = WaveformSet.fromfile(infile)
	f2c = wset.f2c

	if atimes: mtime = np.mean(atimes.values())

	if bpass is not None:
		if not 1 < len(bpass) < 4:
			raise ValueError('Argument bpass should be a sequence of 2 or 3 items')
		bpstart, bpend = bpass[:2]
		try: tails = bpass[2]
		except IndexError: tails = 0
		bpass = (bpstart, bpend, tails)

	bswaves = { }
	for rx in wset.rxidx:
		wf = wset.getwaveform(rx, rx, maptids=True)
		if nsamp: wf.nsamp = nsamp

		dwin = wf.datawin

		if bpass: wf = wf.bandpass(*bpass).window(dwin)

		if atimes:
			try: wf = wf.shift(mtime - atimes[rx])
			except KeyError: pass

		bswaves[rx] = Waveform(wf.nsamp + f2c, wf.getsignal(dwin), dwin.start + f2c)

	return bswaves
		


if __name__ == '__main__':
	dwin = None
	nsamp = None
	bpass = None
	cthresh = None
	zeropad = False
	atimes = None
	freq = 20.
	nproc = None
	hidewf = False

	optlist, args = getopt.getopt(sys.argv[1:], 'hw:a:t:f:n:b:p:m')

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
		elif opt[0] == '-b':
			bpass = [int(s, base=10) for s in opt[1].split(':')]
		elif opt[0] == '-n':
			nsamp = int(opt[1])
		elif opt[0] == '-p':
			nproc = int(opt[1])
		elif opt[0] == '-m':
			hidewf = True
		else:
			usage(fatal=True)

	if len(args) < 2:
		print >> sys.stderr, 'ERROR: required arguments missing'
		usage(fatal=True)

	# Load arrival times and convert to samples
	if atimes is not None:
		atimes = getatimes(atimes, freq)

	# Pull the output name and load the input files
	imgname = args.pop(0)

	try: infiles = matchfiles(args)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(fatal=True)


	# Load and filter all input waves in parallel
	pool = multiprocessing.Pool(processes=nproc)
	result = pool.map_async(getbswaves,
			((infile, atimes, bpass, nsamp) for infile in infiles))
	while True:
		try: result = result.get(5)
		except multiprocessing.TimeoutError: pass
		else:
			bswaves = dict(kp for r in result for kp in r.iteritems())
			break

	pool.close()
	pool.terminate()

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
		print 'Will suppress unaligned waveforms'
		for k, v in bswaves.iteritems():
			if k not in atimes: v *= 0

	print 'Will store waveform image to file', imgname
	waves = [(v, atimes[k] if k in atimes else float('nan')) for k, v in sorted(bswaves.iteritems())]
	plotwaves(imgname, waves, dwin, cthresh)
