#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, itertools, numpy as np
import multiprocessing, Queue

from numpy import ma

from itertools import izip

from collections import OrderedDict

from pycwp import process, stats
from habis import trilateration
from habis.habiconf import HabisConfigError, HabisNoOptionError, HabisConfigParser, matchfiles, buildpaths
from habis.formats import WaveformSet, loadkeymat, savekeymat
from habis.sigtools import Waveform


def kmat(s):
	'''
	If s is a string and sv = s.strip() starts with 'keymat:' (case
	insensitive), return the output of habis.formats.loadkeymat(sv[7:]).
	Otherwise, return s.
	'''
	if isinstance(s, basestring):
		sv = s.strip()
		if sv.startswith('keymat:'):
			return loadkeymat(sv[7:])

	return s


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def finddelays(nproc=1, *args, **kwargs):
	'''
	Distribute, among nproc processes, delay analysis for waveforms using
	calcdelays(). All *args and **kwargs, are passed to calcdelays on each
	participating process. This function explicitly sets the "queue",
	"start", "stride", and "delaycache" arguments of calcdelays, so *args
	and **kwargs should not contain these values.

	The delaycache argument is built from an optional file specified in
	cachefile, which should be a three-column text matrix with rows of the
	form [t, r, delay] representing a precomputed delay for
	transmit-receive pair (t, r).
	'''
	if 'queue' in kwargs:
		raise TypeError("Invalid keyword argument 'queue'")
	if 'stride' in kwargs:
		raise TypeError("Invalid keyword argument 'stride'")
	if 'start' in kwargs:
		raise TypeError("Invalid keyword argument 'start'")
	if 'delaycache' in kwargs:
		raise TypeError("Invalid keyword argument 'delaycache'")

	cachefile = kwargs.pop('cachefile', None)

	try:
		# Try to read an existing delay file and convert to a dictionary
		kwargs['delaycache'] = loadkeymat(cachefile, nkeys=2)
	except (KeyError, ValueError, IOError): pass

	# Create a result queue and a dictionary to accumulate results
	queue = multiprocessing.Queue(nproc)
	delays = { }

	# Extend the kwargs to include the result queue
	kwargs['queue'] = queue
	# Extend the kwargs to include the stride
	kwargs['stride'] = nproc

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

	if len(delays):
		# Save the computed delays, if desired
		try: savekeymat(cachefile, delays, fmt='%d %d %16.8f')
		except (ValueError, IOError): pass

	return delays


def calcdelays(datafile, reffile, osamp, start=0, stride=1, **kwargs):
	'''
	Given a datafile containing a habis.formats.WaveformSet, perform
	cross-correlation on every stride-th waveform, starting at index start,
	to identify the delay of the received waveform relative to a reference
	according to

		wave = datafile.getwaveform(i, j, maptids=True)
		delay = wave.delay(reference, osamp=osamp) + datafile.f2c.

	The addition of datafile.f2c adjusts all delays to a common base time.

	The index (i,j) of the WaveformSet is determined by an optional keyword
	argument 'elmap'. When accessing waveforms for delay analysis, the
	transmit element j is mapped to an appropriate transmission number
	according to any transmit-group configuration specified in the
	WaveformSet. Thus, either all transmit elements j must have
	corresponding receive-channel records in the WaveformSet file, an
	explicit group map must be provided (with the optional kwarg 'groupmap'
	described below), or the file must specify no transmit-group
	configuration (i.e., transmissions must be stored in element order).

	The reference waveform is read from reffile using the method
	habis.sigtools.Waveform.fromfile.

	The return value is a dictionary that maps a (t,r) transmit-receive
	index pair to an adjusted delay in samples.

	The optional kwargs are parsed for the following keys:

	* nsamp: Override datafile.nsamp. Useful mainly for bandpass filtering.

	* bandpass: A dictionary of keyword arguments passed to
	  habis.sigtools.Waveform.bandpass() that will filter each waveform
	  prior to further processing.

	* delaycache: A mapping from transmit-receive element pairs (t, r) to a
	  precomputed delay d. If a value exists for a given pair (t, r), the
	  precomputed value will be used in favor of explicit computation.

	* window: A dictionary containing exactly two of the keys 'start',
	  'length' or 'end' that specify integer values corresponding to the
	  start, length or end of a temporal window applied to each waveform
	  before delay analysis. These keys are passed as the 'window' argument
	  to habis.sigtools.Waveform.window().

	  Two additional keys may be included in the dictionary:

	  - tails: A value passed to the 'tails' argument of the method
	    Waveform.window(), and is either 1) an integer half-width of a Hann
	    window applied to each side, or 2) a list consisting of the
	    concatenation of the start-side and end-side window profiles.

	  - relative: A string, passed as the 'relative' argument to
	    Waveform.window(), that is either 'signal' or 'datawin'.
	    Without specifying 'relative', the window start and end are always
	    specified relative to 0 f2c, and the the applied window becomes:

	      start -> start - datafile.f2c
	      end   -> end - datafile.f2c

	    If 'relative' is specified, the acquisition ('signal') window for
	    each signal starts at datafile.f2c and has length datafile.nsamp.
	    Thus, in either 'relative' mode, the effect is to call

	    	datafile[i,j].window(win, **winargs),

	    Where win is the 'window' dictionary with additional keys removed
	    and winargs is a dictionary of only the additional keys.

	* minsnr: A sequence (mindb, noisewin) used to define the minimum
	  acceptable SNR in dB (mindb) by comparing the peak signal amplitude
	  to the minimum standard deviation over a sliding window of width
	  noisewin. SNR for each signal is calculated after application of an
	  optional window. Delays will not be calculated for signals fail to
	  exceed the minimum threshold.

	* peaks: A dictionary of kwargs to be passed to Waveform.isolatepeak
	  for every waveform. An additional 'nearmap' key may be included to
	  specify a mapping between element indices and expected round-trip
	  delays (relative to zero f2c). The waveform (i,j') for a delay entry
	  (j,i) will be windowed about the peak closest to a delay given by

	    0.5 * (nearmap[txelts[j]] + nearmap[rxelts[i]]) - datafile.f2c

	  to a width twice the peak width, with no tails. If the nearmap is not
	  defined for either element (or at all), the nearmap value will be
	  "None" to isolate the dominant peak.

	  *** NOTE: peak windowing is done after overall windowing and after
	  possible exclusion by minsnr. ***

	* groupmap: A mapping from global element indices to (local index,
	  group index) tuples that will be assigned to the "groupmap" property
	  of the loaded WaveformSet. As part of the property assignment, the
	  groupmap is checked for consistency with receive-channel records in
	  the WaveformSet. Assigning the groupmap property allows (and is
	  required for) specification of transmit channels that are not present
	  in the WaveformSet as receive-channel records.

	* elmap: A mapping from desired receive element indices to one or more
	  transmit indices for which arrival times should be computed. A
	  special string value 'backscatter' (the default) is interpreted as

	  	elmap = { i: [i] for i in datafile.rxidx },

	  while a special string value 'block' is interpreted as

		elmap = { i: list(datafile.rxidx) for i in datafile.rxidx }.

	  Whenever waveforms are accessed with WaveformSet.getwaveform(), the
	  maptids kwarg is set to True.

	* compenv: If True, delay analysis will proceed on signal and reference
	  envelopes. If false, delay analysis uses the original signals.

	* queue: If not none, the return list is passed as an argument to
	  queue.put().
	'''
	# Read the data and reference
	data = WaveformSet.fromfile(datafile)
	ref = Waveform.fromfile(reffile)

	# Override the sample count, if desired
	try: nsamp = kwargs.pop('nsamp')
	except KeyError: pass
	else: data.nsamp = nsamp

	# Assign a global group map, if desired
	try: gmap = kwargs.pop('groupmap')
	except KeyError: pass
	else: data.groupmap = gmap

	# Interpret the element map
	elmap = kwargs.pop('elmap', 'backscatter')

	if isinstance(elmap, basestring):
		elmap = elmap.strip().lower()
		if elmap == 'backscatter':
			elmap = { i: [i] for i in data.rxidx }
		elif elmap == 'block':
			elmap = { i: list(data.rxidx) for i in data.rxidx }
		else:
			raise ValueError("Valid string values for optional kwarg "
						"'elmap' are 'backscatter' or 'block'")

	try:
		items = elmap.items()
	except AttributeError:
		raise ValueError("Optional kwarg 'elmap' is not a proper map")

	# Flatten and sort the element map into a sorted list for striding
	ellst = [ ]
	rxelts = set(data.rxidx)
	txelts = set(data.txidx)
	for k, v in sorted(items):
		# Ignore receive elements not in this file
		if k not in rxelts: continue

		# Build a sorted list of desired transmit elements in this file
		try: s = sorted(sv for sv in set(v) if sv in txelts)
		except TypeError: s = [v] if v in txelts else []

		# Store the flattened pair list
		ellst.extend((k, sv) for sv in s)
		# And make sure the mapping is sorted
		elmap[k] = s

	# Use envelopes for delay analysis if desired
	compenv = kwargs.pop('compenv', False)
	if compenv: ref = ref.envelope()

	# Unpack minimum SNR requirements
	minsnr, noisewin = kwargs.pop('minsnr', (None, None))

	try:
		# Pull a copy of the windowing configuration
		window = dict(kwargs.pop('window'))
	except KeyError:
		window = None
	else:
		winargs = { }
		try: winargs = { 'tails': window.pop('tails') }
		except KeyError: pass
		try: winargs['relative'] = window.pop('relative')
		except KeyError: pass

		if winargs.get('relative', None) is None:
			# For absolute windows, compensate start and end for f2c
			try: window['start'] -= data.f2c
			except KeyError: pass
			try: window['end'] -= data.f2c
			except KeyError: pass

	bandpass = kwargs.pop('bandpass', None)

	# Pull the optional peak search criteria
	peaks = kwargs.pop('peaks', None)
	if peaks is not None:
		nearmap = peaks.pop('nearmap', { })

	# Grab an optional delay cache
	delaycache = kwargs.pop('delaycache', { })

	# Grab an optional result queue
	queue = kwargs.pop('queue', None)

	if len(kwargs):
		raise TypeError("Unrecognized keyword argument '%s'" %  kwargs.iterkeys().next())

	# Compute the strided results
	result = { }

	# Cache a receive-channel record for faster access
	hdr, wforms, tmap = None, None, None

	for rid, tid in ellst[start::stride]:
		try:
			# Use a cahced value if possible
			result[(tid, rid)] = delaycache[(tid, rid)]
			continue
		except KeyError: pass

		try:
			# Use a cached receive-channel block if possible
			if hdr.idx != rid: raise ValueError('Force record load')
		except (ValueError, TypeError, AttributeError):
			# Pull out the desired transmit rows for this receive channel
			try: 
				hdr, wforms = data.getrecord(rid, tid=elmap[rid],
						dtype=np.float32, maptids=True)
			except KeyError:
				raise KeyError('Unable to load Rx %d, Tx %s' % (rid, elmap[rid]))
			# Map transmit indices to desired rows
			tmap = dict(reversed(j) for j in enumerate(elmap[rid]))

		# Grab the signal as a Waveform
		sig = Waveform(data.nsamp, wforms[tmap[tid]], hdr.win.start)

		if bandpass is not None:
			# Remove DC bias to reduce Gibbs phenomenon
			sig.debias()
			# Bandpass and crop to original data window
			sig = sig.bandpass(**bandpass).window(sig.datawin)

		if window is not None:
			# Apply a desired restricted window
			sig = sig.window(window, **winargs)

		if minsnr is not None and noisewin is not None:
			if sig.snr(noisewin) < minsnr: continue

		if peaks is not None:
			# Isolate the peak nearest the expected location (if one exists)
			try: exd = 0.5 * (nearmap[tid] + nearmap[rid]) - data.f2c
			except (KeyError, IndexError): exd = None
			try: sig = sig.isolatepeak(exd, **peaks)
			except ValueError: continue

		if compenv: sig = sig.envelope()

		# Compute and record the delay
		result[(tid, rid)] = sig.delay(ref, osamp) + data.f2c

	try: queue.put(result)
	except AttributeError: pass

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

	kwargs = {}

	try:
		# Read all target input lists
		targets = sorted(k for k in config.options(asec) if k.startswith('target'))
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
		nproc = config.get('general', 'nproc', mapper=int,
				failfunc=process.preferred_process_count)
	except Exception as e:
		err = 'Invalid specification of optional nproc in [general]'
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine the sampling period and a global temporal offset
		dt = config.get(ssec, 'period', mapper=float)
		t0 = config.get(ssec, 'offset', mapper=float)
	except Exception as e:
		err = 'Configuration must specify period and offset in [%s]' % ssec
		raise HabisConfigError.fromException(err, e)

	try:
		# Override the number of samples in WaveformSets
		kwargs['nsamp'] = config.get(ssec, 'nsamp', mapper=int)
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional nsamp in [%s]' % ssec
		raise HabisConfigError.fromException(err, e)


	try:
		# Determine the oversampling rate to use when cross-correlating
		osamp = config.get(ssec, 'osamp', mapper=int, default=1)
	except Exception as e:
		err = 'Invalid specification of optional osamp in [%s]' % ssec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine the map of receive elements to transmit elements
		kwargs['elmap'] = config.get(asec, 'elmap', mapper=kmat, 
					checkmap=False, default='backscatter')
	except Exception as e:
		err = 'Invalid specification of optional elmap in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine a global group mapping to use for transmit row selection
		kwargs['groupmap'] = config.get(asec, 'groupmap',
					checkmap=False, mapper=kmat)
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional groupmap in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine the range of elements to use; default to all (as None)
		kwargs['minsnr'] = config.getlist(asec, 'minsnr', mapper=int)
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional minsnr in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine a temporal window to apply before finding delays
		kwargs['window'] = config.get(asec, 'window')
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional window in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine a temporal window to apply before finding delays
		kwargs['bandpass'] = config.get(asec, 'bandpass')
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional bandpass in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine peak-selection criteria
		kwargs['peaks'] = config.get(asec, 'peaks')
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional peaks in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	maskoutliers = config.get(asec, 'maskoutliers', mapper=bool, default=False)
	optimize = config.get(asec, 'optimize', mapper=bool, default=False)
	cachedelay = config.get(asec, 'cachedelay', mapper=bool, default=True)
	kwargs['compenv'] = config.get(asec, 'compenv', mapper=bool, default=False)

	try:
		# Remove the nearmap file key
		guesses = loadkeymat(kwargs['peaks'].pop('nearmap'), scalar=False)
	except IOError as e:
		guesses = None
		print >> sys.stderr, 'WARNING - Ignoring nearmap:', e
	except (KeyError, TypeError, AttributeError):
		guesses = None
	else:
		# Adjust delay time scales
		guesses = { k: (v - t0) / dt for k, v in guesses.iteritems() }

	times = OrderedDict()

	# Process each target in turn
	for i, (target, datafiles) in enumerate(targetfiles.iteritems()):
		if guesses is not None and 'peaks' in kwargs:
			# Set the nearmap for this target to the appropriate column
			nearmap = { k: v[i] for k, v in guesses.iteritems() }
			kwargs['peaks']['nearmap'] = nearmap

		if cachedelay:
			delayfiles = buildpaths(datafiles, extension='delays.txt')
		else:
			delayfiles = [None]*len(datafiles)

		times[target] = dict()

		print 'Finding delays for target %s (%d data files)' % (target, len(datafiles))

		for (dfile, dlayfile) in izip(datafiles, delayfiles):
			kwargs['cachefile'] = dlayfile

			delays = finddelays(nproc, dfile, reffile, osamp, **kwargs)

			# Note the receive channels in this data file
			lrx = set(k[1] for k in delays.iterkeys())

			# Convert delays to arrival times
			delays = { k: v * dt + t0 for k, v in delays.iteritems() }

			if any(dv < 0 for dv in delays.itervalues()):
				raise ValueError('Non-physical, negative delays exist')

			if maskoutliers:
				# Remove outlying values from the delay dictionary
				delays = stats.mask_outliers(delays)

			if optimize:
				# Prepare the arrival-time finder
				atf = trilateration.ArrivalTimeFinder(delays)
				# Compute the optimized times for this data file
				optimes = { (k, k): v for k, v in atf.lsmr() if k in lrx }
			else:
				# Just pass through the desired times
				optimes = delays

			times[target].update(optimes)

	# Build the combined times list
	for tmap in times.itervalues():
		try: rxset.intersection_update(tmap.iterkeys())
		except NameError: rxset = set(tmap.iterkeys())

	if not len(rxset):
		raise ValueError('Different targets have no common receive-channel indices')

	ctimes = { i: [t[i] for t in times.itervalues()] for i in sorted(rxset) }

	# Save the output as a text file
	# Each data file gets a column
	savekeymat(outfile, ctimes, fmt=['%d', '%d'] + ['%16.8f']*len(times))


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Read the configuration file
	try: config = HabisConfigParser(sys.argv[1])
	except:
		print >> sys.stderr, 'ERROR: could not load configuration file %s' % sys.argv[1]
		usage(sys.argv[0])
		sys.exit(1)

	# Call the calculation engine
	try: atimesEngine(config)
	except Exception as e:
		print >> sys.stderr, "Unable to complete arrival-time estimation;", e
