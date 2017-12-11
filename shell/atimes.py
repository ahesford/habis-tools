#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, itertools, numpy as np, os
import multiprocessing, queue as pyqueue

from collections import OrderedDict, defaultdict

from shlex import split as shsplit

from pycwp import process, stats
from habis import trilateration
from habis.habiconf import HabisConfigError, HabisNoOptionError, HabisConfigParser, matchfiles, buildpaths
from habis.formats import WaveformSet, loadkeymat, loadmatlist, savez_keymat
from habis.sigtools import Waveform


def usage(progname):
	print(f'USAGE: {progname} <configuration>', file=sys.stderr)


def finddelays(nproc=1, *args, **kwargs):
	'''
	Distribute, among nproc processes, delay analysis for waveforms using
	calcdelays(). All *args and **kwargs, are passed to calcdelays on each
	participating process. This function explicitly sets the "queue",
	"start", "stride", and "delaycache" arguments of calcdelays, so *args
	and **kwargs should not contain these values.

	The delaycache argument is built from an optional file specified in
	cachefile, which should be a map from transmit-receive pair (t, r) to a
	precomputed delay, loadable with habis.formats.loadkeymat.
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

	# Try to read an existing delay map
	try: kwargs['delaycache'] = loadkeymat(cachefile)
	except (KeyError, ValueError, IOError): pass

	# Create a result queue and a dictionary to accumulate results
	queue = multiprocessing.Queue(nproc)
	delays = { }

	# Extend the kwargs to include the result queue
	kwargs['queue'] = queue
	# Extend the kwargs to include the stride
	kwargs['stride'] = nproc

	# Keep track of skipped waveforms
	skipped = defaultdict(int)

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
			except pyqueue.Empty: pass
			else:
				delays.update(results[0])
				for k, v in results[1].items():
					if v: skipped[k] += v
				responses += 1

		pool.wait()

	if skipped:
		print(f'For file {os.path.basename(args[0])} '
				f'({len(delays)} identfied times):')
		for k, v in skipped.items():
			if v: print(f'    Skipped {v} {k} waveforms')

	if len(delays):
		# Save the computed delays, if desired
		try: savez_keymat(cachefile, delays)
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

	The return value is a 2-tuple containing, first, a dictionary that maps
	a (t,r) transmit-receive index pair to an adjusted delay in samples;
	and, second, a dictionary that maps reasons for skipping waveforms to
	the number of waveforms skipped for that reason.

	The optional kwargs are parsed for the following keys:

	* flipref: A Boolean (default: False) that, when True, causes the
	  refrence waveform to be negated when read.

	* nsamp: Override datafile.nsamp. Useful mainly for bandpass filtering.

	* negcorr: A Boolean (default: False) passed to Waveform.delay as the
	  'negcorr' argument to consider negative cross-correlation.

	* signsquare: Square the waveform and reference amplitudes (multiplying
	  each signal by its absolute value to preserve signs) to better
	  emphasize peaks in the cross-correlation. The squaring is done right
	  after any bandpass filtering, so other parameters that are influence
	  by amplitude (e.g., minsnr, thresholds in peaks) should be altered to
	  account for the squared amplitudes.

	* bandpass: A dictionary of keyword arguments passed to
	  habis.sigtools.Waveform.bandpass() that will filter each waveform
	  prior to further processing.

	* delaycache: A map from transmit-receive element pairs (t, r) to a
	  precomputed delay d. If a value exists for a given pair (t, r), the
	  precomputed value will be used in favor of explicit computation.

	* window: A map that defines the windowing of signals. Keys in the
	  window map may be:

	  - map: A map from transmit-receive pairs (j,i) to a (start, length)
	    tuple that defines the window to apply to waveform (i,j) in the
	    WaveformSet. The start and length should be specified in samples,
	    with start relative to 0 f2c.

	  - default: A map that contains exactly two of the keys 'start',
	    'length' or 'end' describing window parameters (in samples) to
	    apply to any waveform not identified explicitly in the above map.
	    The map may also contain an optional 'relative' key with an
	    associated value of either 'signal' or 'datawin'. If 'relative' is
	    not specified, the window start and end are relative to 0 f2c, so
	    the applied window is actually

	      start -> start - datafile.f2c
	      end   -> end - datafile.f2c

	    If 'relative' is specified, the 'start' and 'end' parameters of the
	    window are not adjusted and the value of the 'relative' key is
	    passed as the 'relative' keyword argument to Waveform.window.

	  - tails: A value passed to the 'tails' argument of the method
	    Waveform.window(), and is either 1) an integer half-width of a Hann
	    window applied to each side, or 2) a list consisting of the
	    concatenation of the start-side and end-side window profiles.

	* minsnr: A sequence (mindb, noisewin) used to define the minimum
	  acceptable SNR in dB (mindb) by comparing the peak signal amplitude
	  to the minimum standard deviation over a sliding window of width
	  noisewin. SNR for each signal is calculated after application of an
	  optional window. Delays will not be calculated for signals fail to
	  exceed the minimum threshold.

	* peaks: A kwargs map to be passed to Waveform.isolatepeak for every
	  waveform. The map must not contain the 'index' keyword. Two
	  additional keys may be provided:

	  - nearmap: A map from transmit-receive indices (j,i) to expected
	    round-trip delays (in samples, relative to 0 f2c). A waveform (i,j)
	    will use the value (nearmap[j,i] - datafile.f2c), if it exists, as
	    the 'index' argument to Waveform.isolatepeak.

	  - neardefault: A scalar default value to use as an expected
	    round-trip delay (in samples, relative to 0 f2c). Thus, if the
	    value nearmap[j,i] does not exist for some waveform (i,j) in the
	    WaveformSet, the value (nearmap[j,i] - datafile.f2c) will be used
	    as the 'index' argument to Waveform.isolatepeak.

	  If no nearmap or neardefault value can be identified, the index will
	  be 'None' to isolate the dominant peak.

	  *** NOTE: peak windowing is done after overall windowing and after
	  possible exclusion by minsnr. ***

	* groupmap: A map from global element indices to (local index,
	  group index) tuples that will be assigned to the "groupmap" property
	  of the loaded WaveformSet. As part of the property assignment, the
	  groupmap is checked for consistency with receive-channel records in
	  the WaveformSet. Assigning the groupmap property allows (and is
	  required for) specification of transmit channels that are not present
	  in the WaveformSet as receive-channel records.

	* elmap: A map, or a list of maps, from desired receive element
	  indices to one or more transmit indices for which arrival times
	  should be computed. If this parameter is a list of maps, the actual
	  map will be the union of all maps. Any map can also be specified by
	  the magic strings 'backscatter', interpreted as

	  	{ i: [i] for i in datafile.rxidx },

	  or 'block', interpreted as

		{ i: list(datafile.rxidx) for i in datafile.rxidx }.

	  The default map is 'backscatter'.

	* queue: If not none, the return values are passed as an argument to
	  queue.put().

	* eleak: If not None, a floating-point value in the range [0, 1) that
	  specifies the maximum permissible fraction of the total signal energy
	  that may arrive before identified arrival times. Any waveform for
	  which the fraction of total energy arriving before the arrival time
	  exceeds eleak will be rejected as unacceptable.

	  Estimates of energy leaks ignore any fractional parts of arrival
	  times. Energy leaks are estimated after any bandpass filtering or
	  windowing. Estimates never consider peak isolation.
	'''
	# Pull a copy of the IMER configuration, if it exists
	imer = dict(kwargs.pop('imer', ()))

	# Read the data and reference
	data = WaveformSet.fromfile(datafile)

	# Read the reference if IMER times are not desired
	if not imer: ref = Waveform.fromfile(reffile)
	else: ref = None
	# Negate the reference, if appropriate
	if kwargs.pop('flipref', False) and ref is not None: ref = -ref

	# Unpack the signsquare argument and flip the reference if necessary
	signsquare = kwargs.pop('signsquare', False)
	if signsquare and ref is not None: ref = ref.signsquare()

	# Override the sample count, if desired
	try: nsamp = kwargs.pop('nsamp')
	except KeyError: pass
	else: data.nsamp = nsamp

	# Assign a global group map, if desired
	try: gmap = kwargs.pop('groupmap')
	except KeyError: pass
	else: data.groupmap = gmap

	# Determine if an energy "leak" threshold is desired
	try:
		eleak = float(kwargs.pop('eleak'))
	except KeyError:
		eleak = None
	else:
		if not 0 <= eleak < 1:
			raise ValueError('Argument eleak must be in range [0, 1)')

	# Interpret the element map
	elmap = kwargs.pop('elmap', 'backscatter')
	if isinstance(elmap, str): elmap = [elmap]

	if not hasattr(elmap, 'items'):
		# Merge a collection of maps
		dmap = defaultdict(list)
		for en, elm in enumerate(elmap):
			if isinstance(elm, str):
				elm = elm.strip().lower()
				if elm == 'backscatter':
					elm = { i: [i] for i in data.rxidx }
				elif elm == 'block':
					elm = { i: list(data.rxidx) for i in data.rxidx }
				else:
					raise ValueError("Invalid magic element map specified '%s'" % elm)

			try: keys = elm.keys()
			except TypeError: raise TypeError('Invalid element map (index %d)' % en)

			for k in keys:
				# v may be a collection or a scalar
				v = elm[k]
				try: dmap[k].extend(v)
				except TypeError: dmap[k].append(v)

		# Replace the map
		elmap = dict(dmap)
		del dmap

	try: keys = elmap.keys()
	except TypeError: raise TypeError('Invalid element map')

	# Flatten and sort the element map into a sorted list for striding
	ellst = [ ]
	rxelts = set(data.rxidx)
	txelts = set(data.txidx)

	for k in sorted(keys):
		# Ignore receive elements not in this file
		if k not in rxelts: continue

		# v is either a collection or a scalar
		v = elmap[k]

		# Build a sorted list of desired transmit elements in this file
		try: s = sorted(sv for sv in set(v) if sv in txelts)
		except TypeError: s = [v] if v in txelts else []

		# Store the flattened pair list
		ellst.extend((k, sv) for sv in s)
		# Write the trimmed, sorted list back to the map
		elmap[k] = s

	# Unpack minimum SNR requirements
	minsnr, noisewin = kwargs.pop('minsnr', (None, None))

	# Pull a copy of the windowing configuration
	window = dict(kwargs.pop('window', ()))

	# Make sure a per-channel map and empty default window exist
	window.setdefault('map', { })
	window.setdefault('default', { })

	# Determine the relative mode in the default window
	if window['default']:
		relative = window['default'].pop('relative', None)
		if not relative:
			# Compensate start and end for f2c in absolute windows
			try: v = max(window['default']['start'] - data.f2c, 0)
			except KeyError: pass
			else: window['default']['start'] = v

			try: v = max(window['default']['end'] - data.f2c, 0)
			except KeyError: pass
			else: window['default']['end'] = v
		else:
			# Move 'relative' argument to top-level map
			window['relative'] = relative

	bandpass = kwargs.pop('bandpass', None)

	# Pull the optional peak search criteria
	peaks = dict(kwargs.pop('peaks', ()))

	if peaks:
		nearmap = peaks.pop('nearmap', { })
		neardefault = peaks.pop('neardefault', None)

	# Determine whether to allow negative correlations
	negcorr = kwargs.pop('negcorr', False)

	# Grab an optional delay cache
	delaycache = kwargs.pop('delaycache', { })

	# Grab an optional result queue
	queue = kwargs.pop('queue', None)

	if len(kwargs):
		raise TypeError("Unrecognized keyword argument '%s'" %  (next(iter(kwargs.keys())),))

	# Compute the strided results
	result = { }

	# Cache a receive-channel record for faster access
	hdr, wforms, tmap = None, None, None

	numneg = 0
	skipped = defaultdict(int)

	for rid, tid in ellst[start::stride]:
		try:
			# Use a cahced value if possible
			result[(tid, rid)] = delaycache[(tid, rid)]
			skipped['cached'] += 1
			continue
		except KeyError: pass

		if not hdr or hdr.idx != rid:
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

		if bandpass:
			# Remove DC bias to reduce Gibbs phenomenon
			sig.debias()
			# Bandpass and crop to original data window
			sig = sig.bandpass(**bandpass).window(sig.datawin)

		if window:
			wargs = { }
			try: wargs['tails'] = window['tails']
			except KeyError: pass

			try:
				# Check for mapped value
				swin = window['map'][tid,rid]
			except KeyError:
				# Look for a default value
				swin = window.get('default', None)
				try: wargs['relative'] = window['relative']
				except KeyError: pass
			else:
				# Compensate data f2c in mapped value
				swin = (swin[0] - data.f2c, swin[1])

			# Apply a desired window
			if swin: sig = sig.window(swin, **wargs)

		if eleak:
			# Calculate cumulative energy in unwindowed waveform
			cenergy = sig.getsignal(sig.datawin, forcecopy=False)
			cenergy = np.cumsum(cenergy**2)

		# Square the signal if desired
		if signsquare: sig = sig.signsquare()

		if minsnr is not None and noisewin is not None:
			if sig.snr(noisewin, rolling=True) < minsnr:
				skipped['low-snr'] += 1
				continue

		if imer:
			# Compute IMER and its mean
			smsig = sig.imer(**imer).getsignal(sig.datawin, forcecopy=False)
			smlev = np.mean(smsig)
			if smlev < 0:
				smlev = -smlev
				smsig = -smsig
			# Find the first point where IMER breaks the mean
			try: dl = np.nonzero(smsig >= smlev)[0][0]
			except IndexError:
				skipped['failed-IMER'] += 1
				continue
			if dl > 0:
				# Linearly interpolate in the interval if possible
				v0, v1 = smsig[dl - 1], smsig[dl]
				if v0 != v1: dl += (smlev - v1) / (v1 - v0)
			dl += sig.datawin.start
		else:
			if peaks:
				# Isolate peak nearest expected location (if one exists)
				try: exd = nearmap[tid,rid] - data.f2c
				except KeyError:
					if neardefault is None: exd = None
					else: exd = neardefault - data.f2c
				try: sig = sig.isolatepeak(exd, **peaks)[0]
				except ValueError:
					skipped['missing-peak'] += 1
					continue

			# Compute and record the delay
			dl = sig.delay(ref, osamp, negcorr)
			if negcorr:
				if dl[1] < 0: numneg += 1
				dl = dl[0]

		if eleak:
			# Evaluate leaked energy
			ssamp = int(dl) - sig.datawin.start - 1
			if not 0 <= ssamp < len(cenergy):
				skipped['out-of-bounds'] += 1
				continue
			elif cenergy[ssamp] >= eleak * cenergy[-1]:
				skipped['leaky'] += 1
				continue

		result[(tid, rid)] = dl + data.f2c

	if negcorr and numneg:
		print(f'{numneg} waveforms matched with negative cross-correlations')

	try: queue.put((result, skipped))
	except AttributeError: pass

	return result, skipped


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
		elmap = config.get(asec, 'elmap', default='magic:backscatter')

		# Wrap a single value as a 1-element list
		if isinstance(elmap, str): elmap = [elmap]

		# A map should not be further touched, but a list needs loading
		# A simple string is either a magic key or a key matrix
		if not hasattr(elmap, 'keys'):
			def loader(e):
				if e.startswith('magic:'): return e[6:]
				return loadkeymat(e, dtype=int)
			elmap = [ loader(e) for e in elmap ]
		kwargs['elmap'] = elmap
	except Exception as e:
		err = 'Invalid specification of optional elmap in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine a global group mapping to use for transmit row selection
		kwargs['groupmap'] = config.get(asec, 'groupmap')

		# Treat a string groupmap as a file name
		if isinstance(kwargs['groupmap'], str):
			kwargs['groupmap'] = loadkeymat(kwargs['groupmap'], dtype=int)
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
		# Determine an energy leakage threshold
		kwargs['eleak'] = config.get(asec, 'eleak', mapper=float)
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional eleak in [%s]' % asec
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

	try:
		# Determine IMER criteria
		kwargs['imer'] = config.get(asec, 'imer')
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional imer in [%s]' % asec
		raise HabisConfigError.fromException(err, e)


	maskoutliers = config.get(asec, 'maskoutliers', mapper=bool, default=False)
	optimize = config.get(asec, 'optimize', mapper=bool, default=False)
	kwargs['negcorr'] = config.get(asec, 'negcorr', mapper=bool, default=False)
	kwargs['signsquare'] = config.get(asec, 'signsquare', mapper=bool, default=False)
	kwargs['flipref'] = config.get(asec, 'flipref', mapper=bool, default=False)

	# Check for delay cache specifications as boolean or file suffix
	cachedelay = config.get(asec, 'cachedelay', default=True)
	if isinstance(cachedelay, bool) and cachedelay: cachedelay = 'delays.npz'

	try:
		# Remove the nearmap file key
		guesses = shsplit(kwargs['peaks'].pop('nearmap'))
		guesses = loadmatlist(guesses, nkeys=2, scalar=False)
	except IOError as e:
		guesses = None
		print(f'WARNING - Ignoring nearmap: {e}', file=sys.stderr)
	except (KeyError, TypeError, AttributeError):
		guesses = None
	else:
		# Adjust delay time scales
		guesses = { k: (v - t0) / dt for k, v in guesses.items() }

	# Adjust the delay time scales for the neardefault, if provided
	try: v = kwargs['peaks']['neardefault']
	except KeyError: pass
	else: kwargs['peaks']['neardefault'] = (v - t0) / dt

	try:
		# Load the window map, if provided
		winmap = shsplit(kwargs['window'].pop('map'))
		winmap = loadmatlist(winmap, nkeys=2, scalar=False)
	except IOError as e:
		winmap = None
		print(f'WARNING - Ignoring window map: {e}', file=sys.stderr)
	except (KeyError, TypeError, AttributeError):
		winmap = None
	else:
		# Replace the map argument with the loaded array
		kwargs['window']['map'] = winmap

	times = OrderedDict()

	# Process each target in turn
	for i, (target, datafiles) in enumerate(targetfiles.items()):
		if guesses:
			# Pull the column of the nearmap for this target
			nearmap = { k: v[i] for k, v in guesses.items() }
			kwargs['peaks']['nearmap'] = nearmap

		if cachedelay:
			delayfiles = buildpaths(datafiles, extension=cachedelay)
		else:
			delayfiles = [None]*len(datafiles)

		times[target] = dict()

		dltype = 'IMER' if kwargs.get('imer', None) else 'cross-correlation'
		print(f'Finding {dltype} delays for {target} ({len(datafiles)} files)')

		for (dfile, dlayfile) in zip(datafiles, delayfiles):
			kwargs['cachefile'] = dlayfile

			delays = finddelays(nproc, dfile, reffile, osamp, **kwargs)

			# Note the receive channels in this data file
			lrx = set(k[1] for k in delays.keys())

			# Convert delays to arrival times
			delays = { k: v * dt + t0 for k, v in delays.items() }

			if any(dv < 0 for dv in delays.values()):
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
	for tmap in times.values():
		try: rxset.intersection_update(tmap.keys())
		except NameError: rxset = set(tmap.keys())

	if not len(rxset):
		raise ValueError('Different targets have no common receive-channel indices')

	# Cast to Python float to avoid numpy dependencies in pickled output
	ctimes = { i: [float(t[i]) for t in times.values()] for i in sorted(rxset) }

	# Save the output as a pickled map
	savez_keymat(outfile, ctimes)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Read the configuration file
	try: config = HabisConfigParser(sys.argv[1])
	except:
		print(f'ERROR: could not load configuration file {sys.argv[1]}', file=sys.stderr)
		usage(sys.argv[0])
		sys.exit(1)

	# Call the calculation engine
	try: atimesEngine(config)
	except Exception as e:
		print(f'Unable to complete arrival-time estimation: {e}', file=sys.stderr)
