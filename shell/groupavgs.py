#!/opt/python-2.7.9/bin/python

import sys, itertools, ConfigParser, math, numpy as np
import multiprocessing, Queue

from scipy.stats import linregress

from collections import defaultdict

from pycwp import process
from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.formats import WaveformSet
from habis.sigtools import Waveform


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def groupavg(infile, rxchans, grouplen, queue=None, osamp=16):
	'''
	For the habis.formats.WaveformSet object stored in infile, compute the
	average response for each "group" of transmit-receive pairs. A group of
	channels is defined as the set of all channels such that, for each
	channel index I in the set, the ratio (I / grouplen) in integer
	arithmetic, dubbed the group index, takes the same value.

	For each receive group represented in rxchans, a single average
	response is computed by summing all aligned and equalized waveforms
	transmitted from the matching transmit group and measured on all
	receive channels in rxchans belonging to the group. The waveforms are
	aligned by cross-correlation to an arbitrary waveform in the collection
	participating in the average. When cross-correlating, the waveforms are
	oversampled by a factor osamp for sub-wavelength precision.
	Equalization is performed by scaling each waveform by the inverse of
	its peak envelope amplitude (defined as the maximum of the magnitude of
	the Hilbert transform of the wave) so that each waveform has a unit
	peak amplitude.

	All waveforms are expanded to the time scale [0, nsamp], where nsamp is
	the value of the 'nsamp' attribute of the WaveformSet stored in infile.

	The averages for each group are not divided by the number of receive
	channels involved in the calculation, but are divided by the number of
	transmit channels in the group.

	A dictionary whose keys are integer receive-group indices and values
	are corresponding average waveforms is returned. If queue is provided,
	the dictionary is also provided to queue.put() if possible.
	'''
	wset = WaveformSet.fromfile(infile)

	# Map transmit and receive group indices to channel indices
	# Use only local receive channels, but all transmit channels
	rxgroups = defaultdict(list)
	for rx in rxchans:
		rxgroups[int(rx / grouplen)].append(rx)

	txgroups = defaultdict(list)
	for tx in wset.txidx:
		txgroups[int(tx / grouplen)].append(tx)

	# Build a dictionary of average responses
	avgs = {}

	for rxgrp, rxlist in rxgroups.iteritems():
		# Try to find any transmit channels in this group
		try: txlist = txgroups[rxgrp]
		except KeyError: continue

		# Find the number of transmit channels in this grpu
		ntx = len(txlist)

		# Store a running average (empty by default)
		ref = Waveform(wset.nsamp)

		for rxi in rxlist:
			for txi in txlist:
				# Grab the signal waveform
				sig = wset.getwaveform(rxi, txi)
				# Align the signal to reference and convert
				sig = sig.aligned(ref, osamp=osamp, dtype=np.float32)
				# Scale waveform for averaging
				sig /= np.max(sig.envelope()) * float(ntx)
				# Add the scaled waveform to the average
				ref += sig

		# Store the average for this group
		avgs[rxgrp] = ref

	# Put the results on the queue, if desired
	try: queue.put(avgs)
	except AttributeError: pass

	return avgs


def mpgroupavg(infile, grouplen, nproc, osamp=16, regress=None):
	'''
	Subdivide, along receive channels, the work of groupavg() among
	nproc processes to compute average responses for every transmit-receive
	group consisting of grouplen consecutively indexed elements.

	If regress is not None, it should be a tuple of two integers specifying
	the start and end R2C-DFT bin indices over which linear regression will
	be used to determine the phase-angle slope. The waveform will be
	shifted so the slope is near zero. If regress is None, do not shift
	averages waveforms from their arbitrary positions.
	'''
	# Grab the receive channels to be distributed
	rxidx = WaveformSet.fromfile(infile).rxidx

	# Create a queue to receive per-process results
	queue = multiprocessing.Queue(nproc)

	# Collect lists of per-process group averages
	avglists = defaultdict(list)

	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Assign a meaningful process name
			procname = process.procname(i)
			# Block the receive channels
			share, rem = len(rxidx) / nproc, len(rxidx) % nproc
			start = i * share + min(i, rem)
			share = share + int(i < rem)
			args = (infile, rxidx[start:start+share], grouplen, queue, osamp)
			pool.addtask(target=groupavg, name=procname, args=args)

		pool.start()

		responses = 0
		while responses < nproc:
			try:
				# Collect the per-process averages, grouping by group index
				results = queue.get(timeout=0.1)
				for key, value in results.iteritems():
					avglists[key].append(value)
				responses += 1
			except Queue.Empty: pass

		pool.wait()

	# Compute the overall group averages
	averages = {}
	for gidx, avgs in avglists.iteritems():
		ref = avgs[0]
		for avg in avgs[1:]:
			ref += avg.aligned(ref, osamp=osamp)
		if regress is None:
			averages[gidx] = ref
		else:
			# Compute the DFT frequencies
			rstart, rend = regress
			kidx = np.arange(rstart, rend)
			freqs = -2.0 * math.pi * kidx / ref.nsamp
			# Compute the phase angle and regress
			ang = np.unwrap(np.angle(ref.fft())[rstart:rend])
			slope = linregress(freqs, ang)[0]
			# Shift out the slope
			averages[gidx] = ref.shift(-slope)

	return averages


def averageEngine(config):
	'''
	With multiple processes, perform bandpass filtering on a configured
	habis.formats.WaveformSet stored in a file and then compute average
	responses for successive groups of contiguous blocks of channels.
	'''
	try:
		datafiles = config.getlist('average', 'datafile')
		outformats = config.getlist('average', 'outformat')
		if len(datafiles) < 1 or len(datafiles) != len(outformats):
			raise ConfigParser.Error('Fall-through to exception handler')
	except ConfigParser.Error:
		raise HabisConfigError('Configuration must specify datafile and corresponding outformat in [average]')

	try:
		osamp = config.getint('average', 'osamp')
	except:
		raise HabisConfigError('Invalid specification of osamp in [average]')

	try:
		nproc = config.getint('general', 'nproc')
	except ConfigParser.NoOptionError:
		nproc = process.preferred_process_count()
	except:
		raise HabisConfigError('Invalid specification of process count in [general]')

	try:
		regress = config.getlist('average', 'regress', int)
		if len(regress) != 2:
			raise ValueError('Fall-through to exception handler')
	except ConfigParser.NoOptionError:
		regress = None
	except:
		raise HabisConfigError('Specification of optional regress in [average] must contain two ints')

	try:
		grouplen = config.getint('average', 'grouplen')
	except:
		raise HabisConfigError('Invalid specification of grouplen in [average]')

	for datafile, outformat in zip(datafiles, outformats):
		print 'Computing average responses for data file', datafile
		avgs = mpgroupavg(datafile, grouplen, nproc, osamp, regress)
		for gidx, avg in avgs.iteritems():
			outfile = outformat.format(gidx)
			avg.store(outfile)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Read the configuration file
	try: config = HabisConfigParser.fromfile(sys.argv[1])
	except:
		print >> sys.stderr, 'ERROR: could not load configuration file %s' % sys.argv[1]
		usage(sys.argv[0])
		sys.exit(1)

	# Call the calculation engine
	averageEngine(config)
