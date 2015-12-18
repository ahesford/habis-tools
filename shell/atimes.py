#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, itertools, numpy as np
import multiprocessing, Queue

from numpy import ma

from itertools import izip

from collections import OrderedDict

from pycwp import process, cutil
from habis import trilateration
from habis.habiconf import HabisConfigError, HabisConfigParser, matchfiles, buildpaths
from habis.formats import WaveformSet
from habis.sigtools import Waveform


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def validatechans(wset, kwargs):
	'''
	Validate the contents of the 'rxelts' and 'txelts' sequences in the
	kwargs dictionary, if they exist. If rxelts is not specified, populate
	it with sorted(wset.rxidx). If txelts is not specified, populate it
	with rxelts. Returns (rxelts, txelts).
	'''
	# Pull the rxelts list
	rxelts = kwargs.get('rxelts', None)

	if rxelts is None: rxelts = sorted(wset.rxidx)
	else: rxelts = sorted(set(rxelts).intersection(wset.rxidx))

	# Pull the txelts list
	txelts = kwargs.get('txelts', None)
	if txelts is None: txelts = list(rxelts)

	if not set(txelts).issubset(wset.txidx):
		raise ValueError('Transmit-channel list contains elements not in data file')

	return rxelts, txelts


def finddelays(datafile, delayfile=None, nproc=1, *args, **kwargs):
	'''
	Compute the delay matrix for a habis.formats.WaveformSet stored in
	datafile. If delayfile is specified and has the form

		[[t, r, delay] for t in txelts for r in rxelts],

	the delays are read from that file. Otherwise, the WaveformSet is read
	directly and delays are determined through delay analysis.

	Delay analysis for individual waveforms is farmed out to calcdelays()
	among nproc processes. The datafile argument, along with *args and
	**kwargs, are passed to calcdelays on each participating process. This
	function explicitly sets the "queue", "start", and "stride" arguments
	of calcdelays, so *args and **kwargs should not contain these values.

	If delayfile is specified but computation is still required, the
	computed delays will be saved to delayfile. Any existing content will
	be overwritten.
	'''
	# Grab the participating channels
	rxelts, txelts = validatechans(WaveformSet.fromfile(datafile), kwargs)

	try:
		# Try to read an existing delay file and convert to a dictionary
		delays = {(int(t), int(r)): v for (t, r, v) in np.loadtxt(delayfile)}
		# Ensure the cache contains exactly the right keys
		keys = set(delays.iterkeys())
		for t, r in itertools.product(txelts, rxelts):
			try: keys.remove((t,r))
			except KeyError: raise ValueError('Missing delay in cache')
		if len(keys): raise ValueError('Extra delay in cache')
		return delays
	except (ValueError, IOError):
		# ValueError if format is inappropriate or delayfile is None
		# IOError if delayfile does not point to an existent file
		pass

	# Create a result queue and a dictionary to accumulate results
	queue = multiprocessing.Queue(nproc)
	delays = { }

	# Extend the kwargs to include the result queue
	kwargs['queue'] = queue
	# Extend the kwargs to include the stride
	kwargs['stride'] = nproc
	# Insert the datafile at the beginning of the args
	args = tuple([datafile] + list(args))

	# Spawn the desired processes to perform the cross-correlation
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Pick a useful process name
			procname = process.procname(i)
			# Add the start index to the kwargs
			kwargs['start'] = i
			# Extend kwargs to contain the queue (copies kwargs)
			pool.addtask(target=calcdelays, name=procname, args=args, kwargs=kwargs)

		pool.start()

		# Wait for all processes to respond
		responses = 0
		while responses < nproc:
			try: results = queue.get(timeout=0.1)
			except Queue.Empty: pass
			else:
				delays.update(results)
				responses += 1

		pool.wait()

	try:
		# Save the computed delays, if desired
		np.savetxt(delayfile,
				[[t, r, v] for (t, r), v in delays.iteritems()],
				fmt='%d %d %16.8f')
	except (ValueError, IOError):
		pass

	return delays


def calcdelays(datafile, reffile, osamp, start=0, stride=1, **kwargs):
	'''
	Given a datafile containing a habis.formats.WaveformSet, perform
	cross-correlation on every stride-th waveform, starting at index start,
	to identify the delay of the received waveform relative to a reference
	according to

		datafile[i,j].delay(reference, osamp=osamp) + datafile.f2c.

	The addition of datafile.f2c adjusts all delays to a common base time.

	The index (i,j) of the WaveformSet is determined by un-flattening the
	strided index, k, into a 2-D index (j,i) into the T x R delay matrix in
	row-major order.

	The reference waveform is read from reffile using
	habis.sigtools.Waveform.fromfile.

	The return value is a dictionary that maps a (t,r) transmit-receive
	index pair to an adjusted delay in samples.

	The optional kwargs are parsed for the following keys:

	* window: If not None, should be a (start, length, [tailwidth]) tuple
	  of ints that specifies a temporal window applied to each waveform
	  before delay analysis. The start should be relative to 0
	  fire-to-capture delay; the actual window used will be

	  	(start - datafile.f2c, length, [tailwidth]).

	  The optional tailwidth specifies the half-width of a Hann window
	  passed as the tails argument to Waveform.window.

	* peaks: If not None, should be a dictionary of kwargs to be passed to
	  Waveform.envpeaks for every waveform. Additionally, a 'nearmap' key
	  must be included to specify a list of expected round-trip delays
	  (relative to zero f2c) for each element. The waveform (i,j) for a
	  delay entry (j,i) will be windowed about the peak closest to a delay
	  given by
	  
	    0.5 * (nearmap[txelts[j]] + nearmap[rxelts[i]]) - datafile.f2c
	  
	  to a width twice the peak width, with no tails. Note: peak windowing
	  is done after overall windowing.

	* rxelts and txelts: If not None, should be a lists of element indices
	  such that entry (i,j) in the delay matrix corresponds to the waveform
	  datafile[rxelts[j],txelts[i]]. When rxelts is None or unspecified, it
	  is populated by sorted(datafile.rxidx). When txelts is None or
	  unspecified, it defaults to rxelts.

	* compenv: If True, delay analysis will proceed on signal and reference
	  envelopes. If false, delay analysis uses the original signals.

	* queue: If not none, the return list is passed as an argument to
	  queue.put().
	'''
	# Read the data and reference
	data = WaveformSet.fromfile(datafile)
	ref = Waveform.fromfile(reffile)

	# Determine the elements to include in the delay matrix
	rxelts, txelts = validatechans(data, kwargs)

	# Use envelopes for delay analysis if desired
	compenv = kwargs.get('compenv', False)
	if compenv: ref = ref.envelope()

	t, r = len(txelts), len(rxelts)

	# Pull the window and compute optional tails
	window = kwargs.get('window', None)
	try:
		window = list(window)
	except TypeError:
		pass
	else:
		window[0] -= data.f2c
		try: tails = np.hanning(2 * window[2])
		except IndexError: tails = None

	# Pull the optional peak search criteria
	peaks = kwargs.get('peaks', None)
	try: nearmap = peaks.pop('nearmap', None)
	except AttributeError: nearmap = None

	if peaks is not None and nearmap is None:
		raise KeyError('kwarg "peaks" must be a dictionary with a "nearmap" key')

	# Compute the strided results
	result = { }
	for idx in range(start, t * r, stride):
		# Find the transmit and receive indices
		i, j = np.unravel_index(idx, (t, r), 'C')
		tid, rid = txelts[i], rxelts[j]
		# Pull the waveform as float32
		sig = data.getwaveform(rid, tid, dtype=np.float32)
		if window: sig = sig.window(window[:2], tails=tails)
		if peaks:
			# Isolate the peak nearest the expected location
			exd = 0.5 * (nearmap[tid] + nearmap[rid]) - data.f2c
			sig = sig.isolatepeak(exd, **peaks)
		if compenv: sig = sig.envelope()
		# Compute and record the delay
		result[(tid, rid)] = sig.delay(ref, osamp) + data.f2c

	try: kwargs['queue'].put(result)
	except (KeyError, AttributeError): pass

	return result


def atimesEngine(config):
	'''
	Use habis.trilateration.ArrivalTimeFinder to determine a set of
	round-trip arrival times from a set of one-to-many multipath arrival
	times. Multipath arrival times are computed as the maximum of
	cross-correlation with a reference pulse, plus some constant offset.
	'''
	asec = 'atimes' 
	msec = 'measurement'
	ssec = 'sampling'
	try:
		# Read all target input lists
		targets = sorted(k for k, v in config.items(asec) if k.startswith('target'))
		targetfiles = OrderedDict()
		for target in targets:
			targetfiles[target] = matchfiles(config.getlist(asec, target))
			if len(targetfiles[target]) < 1:
				err = 'Key %s matches no input files' % target
				raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify at least one unique "target" key in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the reference file
		reffile = config.get(msec, 'reference')
	except Exception as e:
		err = 'Configuration must specify reference in [%s]' % msec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the output file
		outfile = config.get(asec, 'outfile')
	except Exception as e:
		err = 'Configuration must specify outfile in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the number of processes to use (optional)
		nproc = config.getint('general', 'nproc',
				failfunc=process.preferred_process_count)
	except Exception as e:
		err = 'Invalid specification of optional nproc in [general]'
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine the sampling period and a global temporal offset
		dt = config.getfloat(ssec, 'period')
		t0 = config.getfloat(ssec, 'offset')
	except Exception as e:
		err = 'Configuration must specify period and offset in [%s]' % ssec
		raise HabisConfigError.fromException(err, e)

	try:
		osamp = config.getint(ssec, 'osamp', failfunc=lambda: 1)
	except Exception as e:
		err = 'Invalid specification of optional osamp in [%s]' % ssec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine the range of elements to use; default to all (as None)
		txelts = config.getrange(asec, 'txelts', failfunc=lambda: None)
		rxelts = config.getrange(asec, 'rxelts', failfunc=lambda: None)
	except Exception as e:
		err = 'Invalid specification of optional txelts, rxelts in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine a temporal window to apply before finding delays
		window = config.getlist(asec, 'window',
				mapper=int, failfunc=lambda: None)
		if window and (len(window) < 2 or len(window) > 3):
			err = 'Window does not specify appropriate parameters'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Invalid specification of optional window in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine peak-selection criteria
		peaks = config.getlist(asec, 'peak', failfunc=lambda: None)
		if peaks:
			if len(peaks) < 2 or len(peaks) > 5:
				err = 'Peak does not specify appropriate parameters'
				raise HabisConfigError(err)
			if peaks[0].lower() != 'nearest':
				err = 'Peak specification must start with "nearest"'
				raise HabisConfigError(err)
			# Build the peak-selection options dictionary
			peakargs = { 'nearfile': peaks[1] }
			if len(peaks) > 2:
				peakargs['minwidth'] = float(peaks[2])
			if len(peaks) > 3:
				peakargs['minprom'] = float(peaks[3])
			if len(peaks) > 4:
				peakargs['prommode'] = peaks[4]
			peaks = peakargs
	except Exception as e:
		err = 'Invalid specification of optional peak in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	usediag = config.getboolean(asec, 'usediag', failfunc=lambda: False)
	maskoutliers = config.getboolean(asec, 'maskoutliers', failfunc=lambda: False)
	compenv = config.getboolean(asec, 'compenv', failfunc=lambda: False)
	cachedelay = config.getboolean(asec, 'cachedelay', failfunc=lambda: True)

	try:
		# If a delay guess was specified, read the delay matrix
		# Also remove the 'nearfile' key
		guesses = np.loadtxt(peaks.pop('nearfile'))
		# Convert from times to samples
		guesses = (guesses - t0) / dt
	except (KeyError, TypeError, IOError, AttributeError):
		guesses = None

	times = OrderedDict()

	# Process each target in turn
	for i, (target, datafiles) in enumerate(targetfiles.iteritems()):
		# Set the peaks nearmap to the appropriate guess column
		try: peaks['nearmap'] = guesses[:,i]
		except TypeError: pass

		if cachedelay:
			delayfiles = buildpaths(datafiles, extension='delays.txt')
		else:
			delayfiles = [None]*len(datafiles)

		times[target] = dict()

		for (dfile, dlayfile) in izip(datafiles, delayfiles):
			print 'Finding delays for data set', dfile

			delays = finddelays(dfile, dlayfile, nproc, reffile,
					osamp, compenv=compenv, window=window,
					peaks=peaks, rxelts=rxelts, txelts=txelts)

			# Note the receive channels in this data file
			lrx = set(k[1] for k in delays.iterkeys())

			# Convert delays to arrival times
			delays = { k: v * dt + t0 for k, v in delays.iteritems() }

			if np.any(delays.itervalues() < 0):
				raise ValueError('Non-physical, negative delays exist')

			if maskoutliers:
				# Remove outlying values from the delay dictionary
				q1, q2, q3 = numpy.percentile(delays.values(), [25, 50, 75])
				iqr = q3 - q1
				lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
				delays = { k: v for k, v in delays.iteritems() if lo <= v <= hi }

			if not usediag:
				# Prepare the arrival-time finder
				atf = trilateration.ArrivalTimeFinder(delays)
				# Compute the optimized times for this data file
				optimes = { k: v for k, v in atf.lsmr() if k in lrx }
			else:
				# Take only diagonal values
				optimes = { k[0]: v for k, v in delays.iteritems() if k[0] == k[1] }

			times[target].update(optimes)

	# Build the combined times list
	for tmap in times.itervalues():
		try: rxset.intersection_update(tmap.iterkeys())
		except NameError: rxset = set(tmap.iterkeys())

	if not len(rxset):
		raise ValueError('Different targets have no common receive-channel indices')

	ctimes = [[i] + [t[i] for t in times.itervalues()] for i in sorted(rxset)]

	# Save the output as a text file
	# Each data file gets a column
	np.savetxt(outfile, ctimes, fmt=' '.join(['%d'] + ['%16.8f'] * len(times)))


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
	atimesEngine(config)
