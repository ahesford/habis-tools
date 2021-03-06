#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, itertools, numpy as np, os
import multiprocessing, queue as pyqueue

from numpy.linalg import norm

from collections import OrderedDict, defaultdict

from shlex import split as shsplit

from pycwp import process, stats, cutil, signal
from habis import trilateration
from habis.habiconf import HabisConfigError, HabisNoOptionError, HabisConfigParser, matchfiles, buildpaths
from habis.formats import loadkeymat, loadmatlist, savez_keymat, strict_nonnegative_int
from habis.sigtools import Waveform, Window, WaveformMap

def usage(progname):
	print(f'USAGE: {progname} <configuration>', file=sys.stderr)

def finddelays(nproc=1, *args, **kwargs):
	'''
	Distribute, among nproc processes, delay analysis for waveforms using
	calcdelays(). All *args and **kwargs, are passed to calcdelays on each
	participating process. This function explicitly sets the "queue",
	"rank", "grpsize", and "delaycache" arguments of calcdelays, so *args
	and **kwargs should not contain these values.

	The delaycache argument is built from an optional file specified in
	cachefile, which should be a map from transmit-receive pair (t, r) to a
	precomputed delay, loadable with habis.formats.loadkeymat.
	'''
	forbidden = { 'queue', 'rank', 'grpsize', 'delaycache' }
	forbidden.intersection_update(kwargs)
	if forbidden:
		raise TypeError("Forbidden argument '{next(iter(forbidden))}'")

	cachefile = kwargs.pop('cachefile', None)

	# Try to read an existing delay map
	try: kwargs['delaycache'] = loadkeymat(cachefile)
	except (KeyError, ValueError, IOError): pass

	# Create a result queue and a dictionary to accumulate results
	queue = multiprocessing.Queue(nproc)
	delays = { }

	# Extend the kwargs to include the result queue
	kwargs['queue'] = queue
	# Extend the kwargs to include the group size
	kwargs['grpsize'] = nproc

	# Keep track of waveform statistics
	stats = defaultdict(int)

	# Spawn the desired processes to perform the cross-correlation
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Pick a useful process name
			procname = process.procname(i)
			# Add the group rank to the kwargs
			kwargs['rank'] = i
			# Extend kwargs to contain the queue (copies kwargs)
			pool.addtask(target=calcdelays, name=procname, args=args, kwargs=kwargs)

		pool.start()

		# Wait for all processes to respond
		responses, deadpool = 0, False
		while responses < nproc:
			try: results = queue.get(timeout=0.1)
			except pyqueue.Empty:
				# Loosely join to watch for a dead pool
				pool.wait(timeout=0.1, limit=1)
				if not pool.unjoined:
					# Note a dead pool, give read one more try
					if deadpool: break
					else: deadpool = True
			else:
				delays.update(results[0])
				for k, v in results[1].items():
					if v: stats[k] += v
				responses += 1

		if responses != nproc:
			print(f'WARNING: Proceeding with {responses} of {nproc} '
					'subprocess results. A subprocess may have died.')

		pool.wait()

	if stats:
		print(f'For file {os.path.basename(args[0])} '
				f'({len(delays)} identfied times):')
		for k, v in sorted(stats.items()):
			if v:
				wfn = 'waveforms' if v > 1 else 'waveform'
				print(f'  {v} {k} {wfn}')

	if len(delays) and cachefile:
		# Save the computed delays, if desired
		try: savez_keymat(cachefile, delays)
		except (ValueError, IOError): pass

	return delays


def isolatepeak(sig, key, nearmap={ }, neardefault=None, **kwargs):
	'''
	For the signal sig, isolate a peak by calling

		sig.isolatepeak(exd, **kwargs),

	were exd is nearmap[key] if such a value exists or else is neardefault.

	Only the first return value of sig.isolatepeak (the isolated signal) is
	returned.

	A ValueError will be raised if a peak cannot be isolated.
	'''
	return sig.isolatepeak(nearmap.get(key, neardefault), **kwargs)[0]


def getimertime(sig, osamp=1, threshold=1, window=None, equalize=True,
		rmsavg=None, absimer=False, merpeak=False,
		breakaway=None, breaklen=None, interpolate=True, **kwargs):
	'''
	For the signal sig, compute the IMER function imer = sig.imer(**kwargs)
	or, if absimer is True, imer = abs(sig.imer(**kwargs)) and find the
	index of the first sample that is at least as large as:

		threshold * mean(imer), if rmsavg is None, or else
		threshold * sqrt(RMS * max(abs(imer))),

	where RMS is the RMS value of the IMER function from the beginning of
	time to the sample argmax(abs(imer)) - int(rmsavg). If the RMS window
	is not well defined (i.e., the value of rmsavg moves the end of the
	window before the start of the IMER data window), the threshold search
	will proceed as if rmsavg were None.

	The parameter osamp, if osamp != 1, is used to oversample the signal as
	sig.oversample(osamp) before any analysis is performed. Thus, all
	parameters that depend on sample values (breaklen, window  and IMER
	kwargs) will be scaled by osamp as well.

	If window is None, the whole-signal IMER is searched. Otherwise, window
	should be a tuple or dictionary that will be convered to a Window
	object, as Window(**window) if possible or as Window(*window)
	otherwise, that will be used to limit the evaluation of the mean and
	the search for a threshold crossing.

	If equalize is True, the signal is equalized by dividing by the peak
	envelope amplitude within the desired window prior to any IMER search.

	If a primary crossing is found and breakaway is not None, a secondary
	search is performed to identify the latest sample of the IMER function
	before the primary crossing that crosses the secondary threshold

		secthresh = mean(imer[secstart:secondary]) +
				breakaway * std(imer[secstart:secondary]),

	where secondary is the index being tested for a secondary crossing and
	secstart is 0 if breaklen is None and (secondary - breaklen) otherwise.
	If a suitable secondary crossing is found, the IMER time is replaced by
	the secondary crossing.

	The breaklen argument is ignored if breakaway is None.

	If a primary crossing is found and merpeak is True, the IMER time will
	be replaced by the location a peak in the near-MER function at least as
	large as the value of the near-MER function at the primary IMER
	crossing and subject to the following rules:

	* If any suitably high near-MER peaks occur before the primary IMER
	  crossing, the highest such peak will be selected, otherwise,

	* If all suitably high near-MER peaks occur no earlier than the primary
	  IMER crossing, the earliest such peak will be selected.
	
	If merpeak is a numeric value, it will be interpreted as an integer and
	the near-MER function will be smoothed with a rolling average of that
	many points prior to the peak search. The rolling average window will
	be centered with respect to the data to preserve peak locations.

	The merpeak and breakaway options are mutually exclusive.

	If interpolate is True, the threshold crossing will be interpolated
	between samples by fitting a cubic spline to points surrounding the
	interval where the threshold crossing was detected. By default, eight
	points (four on either side of the interval, if possible) will be used
	to construct the spline; if interpolate is a positive integer, that
	number of points will be used instead. Interpolation is not supported
	with merpeak values.

	An IndexError will be raised if a suitable primary crossing cannot be
	identified.
	'''
	if threshold < 0: raise ValueError('IMER threshold must be positive')

	if osamp != int(osamp) or osamp < 1:
		raise ValueError('Parameter "osamp" must be a nonnegative integer')

	if merpeak:
		if breakaway is not None:
			raise ValueError('Options "merpeak" and '
					'"breakaway" are mutually exclusive')
		merpeak = int(merpeak)
		if merpeak < 1:
			raise ValueError('Option "merpeak" must be '
					'Boolean or a positive integer')

	if interpolate:
		if merpeak: raise ValueError('Interpolation not supported with merpeak')
		if interpolate is True: interpolate = 8
		else: interpolate = strict_nonnegative_int(interpolate)

	if osamp != 1:
		sig = sig.oversample(osamp)
		if breaklen: breaklen = breaklen * osamp
		# Scale the IMER values
		for key in ['avgwin', 'vpwin', 'prewin', 'postwin']:
			try: kwargs[key] = osamp * kwargs[key]
			except KeyError: pass

	# Figure start and length of interest window WRT data window
	if window is None:
		dstart, dlen = 0, sig.datawin.length
	else:
		try: window = Window(**window)
		except TypeError: window = Window(*window)

		# Scale the window by the oversampling parameter
		if osamp != 1: window = window.oversample(osamp)

		try: dstart, _, dlen = cutil.overlap(sig.datawin, window)
		except TypeError:
			raise IndexError('Specific window does not overlap signal data')

	dend = dstart + dlen

	if equalize:
		# Equalize with peak in the desired window
		if window is None: emax = sig.envelope().extremum()[0]
		else: emax = sig.window(window).envelope().extremum()[0]
		sig /= emax

	if merpeak:
		# Compute the IMER and NMER only in the data window
		imer = sig.imer(raw=True, return_all=True, **kwargs)
		imer, nmer = imer['imer'], imer['nmer']
	else: imer = sig.imer(raw=True, **kwargs)

	# Use the magnitude of IMER if desired
	if absimer: imer = np.abs(imer)

	# Compute primary threshold value in restricted window
	ime = imer[dstart:dend]
	mval = None

	if rmsavg is not None:
		# Find peak location and end of RMS interval
		pk = np.argmax(np.abs(ime))
		rmend = pk + dstart - int(rmsavg)
		if rmend > 0:
			# Find RMS value for all time to start of gap
			rmsval = np.sqrt(np.mean(imer[:rmend]**2))
			# Compute compromise threshold
			mval = threshold * np.sqrt(np.abs(ime[pk]) * rmsval)

	# Default threshold when rmsavg is None or no RMS baseline can be found
	if mval is None: mval = threshold * np.mean(ime)

	# Find the first crossing of the IMER function
	try: dl = np.nonzero(imer[dstart:dend] >= mval)[0][0] + dstart
	except IndexError: raise IndexError('Primary IMER crossing not found')

	# Searches beyond the first sample are useless
	if dl < 1: return dl + sig.datawin.start

	if breakaway is not None:
		# Truncate secondary search to primary crossing
		ime = imer[:dl]

		# Build baseline up to primary crossing, growing from start of IMER
		mustd = stats.rolling_std(ime, breaklen, expand=True, with_mean=True)
		# When with_mean is True, array has mean in column 1, std in column 0
		baseline = mustd[:,1] + breakaway * mustd[:,0]

		# Limit leading edge of IMER and baseline to start of window
		ime = imer[dstart:]
		baseline = baseline[dstart:]

		try:
			# Latest crossing is first point after last below baseline
			dls = np.nonzero(ime < baseline)[0][-1] + 1
			# If secondary crossing runs past primary, no more searching
			if dls + dstart >= dl: raise IndexError
		except IndexError:
			pass
		else:
			# Establish the new threshold from crossing
			mval = baseline[dls]

			# Shift delay to start of data window
			dl = dls + dstart
	elif merpeak:
		if merpeak > 1:
			# Use rolling-average filter to smooth NMER
			nme = stats.rolling_mean(nmer, merpeak, expand=True)
			# Shift by half width to center the rolling average
			hwl = (merpeak - 1) // 2
			lnh = len(nmer) - hwl
			nmer[:lnh] = nme[hwl:]
			# Replicate the last smooth value over final half window
			nmer[lnh:] = nmer[lnh-1]
		# Find the NMER level at the IMER crossing
		nmval = nmer[dl]
		# Find high-enough peaks in window, sort by index
		nmpeaks = sorted(pk['peak']
				for pk in signal.findpeaks(nmer[dstart:dend])
					if pk['peak'][1] > nmval)
		# Find the preferred NMER peak, walking in increasing index order
		bestpk = None
		for idx, height in nmpeaks:
			# Shift peak to position within data window
			idx = idx + dstart
			if idx < dl:
				# For peaks before crossing, pick highest one
				if not bestpk or height > bestpk[1]:
					bestpk = idx, height
			else:
				# Passed IMER crossing; use this peak
				# unless a pre-crossing peak was already found
				if not bestpk: bestpk = idx, height
				break

		# Update arrival time to location of preferred peak
		if bestpk: dl = bestpk[0]

	# Interpolate if desired and possible
	if interpolate:
		from scipy.interpolate import splrep, sproot
		hfi = interpolate // 2
		ibeg = max(0, dl - hfi)
		iend = min(len(imer), ibeg + interpolate)
		# Build a cubic spline representation if enough points exist
		try: tck = splrep(np.arange(ibeg, iend), imer[ibeg:iend] - mval, s=0)
		except TypeError: pass
		else:
			# Replace delay with cubic root in proper interval
			try: dl = next(rt for rt in sproot(tck) if dl - 1 < rt <= dl)
			except StopIteration: pass

	# Offset IMER with the start of the data window and adjust for oversampling
	arrival = dl + sig.datawin.start
	if osamp != 1: arrival /= osamp

	return arrival


def applywindow(sig, key, map={ }, default=None, **kwargs):
	'''
	For a given signal sig, search for a given window map[key]. The map
	should provide values of the form (start, length) that are suitable to
	pass as the first argument to Waveform.window.

	If map[key] does not exist, the default window given by default (if it
	is not None) will be used. The default value can be anything suitable
	as the first argument of Waveform.window. As a special case, if default
	is a dictionary and contains the 'relative' key, the value of that key
	will be passed as the 'relative' keyword argument to Waveform.window.

	If map[key] does not exist and default is not specified, no window will
	be applied, and the returned signal will be identical to sig.

	If a window is identified (either from map[key] or default), the return
	value will be

		sig.window(window, **kwargs)

	where, if 'relative' was specified as a key in 'default' as a
	dictionary, the value of 'relative' will be added to kwargs.
	'''
	try:
		start, length = map[key]
		win = { 'start': start, 'length': length }
	except KeyError:
		if not default: return sig

		# Make a copy of a dictionary default
		if hasattr(default, 'keys'):
			win = dict(default)
		else:
			start, length = default
			win = { 'start': start, 'length': length }

	try: relative = win.pop('relative')
	except KeyError: pass
	else: kwargs['relative'] = relative

	return sig.window(win, **kwargs)


def wavegen(data, neighbors={ }, bandpass=None, window=None, rank=0, grpsize=1):
	'''
	From a WaveformMap data, yield a (t, r) tuple, corresponding Waveform
	object and a set of (t,r) pairs in the neighborhood of each yielded
	waveform.

	The neighborhood of each waveform is defined by the map neighbors,
	which (if not empty) should map each element index to a collection of
	neighboring elements. (The neighbor collection for each index is always
	assumed to contain that index, whether or not the collection includes
	it.) A neighborhood of a pair (t, r) consists of all pairs in the cross
	product of the neighbrhoods of t and r, respectively.

	Of the (t, r) pairs available in data, a local share for the given
	process rank in a process group of size grpsize will be yielded.

	If bandpass is not None, it should be a dictionary suitable for passing
	as keyword arguments (**bandpass) to Waveform.bandpass for each
	waveform pulled from data prior to further processing.

	If window is not None, it should be a dictionary suitable for passing
	as keyword arguments (**window) to applywindow. The first three
	arguments to applywindow are provided by this function and should not
	appear in the window dictionary.
	'''
	# Build a list of neighborhoods over which waveforms will be averaged
	neighborhoods = { }
	emptyset = set()

	for (t, r) in data:
		# Find the receive neighbors
		lrn = {r}.union(neighbors.get(r, emptyset))
		# Find the transmit neighbors
		ltn = {t}.union(neighbors.get(t, emptyset))
		# Build the neighborhood
		nbrs = set(itertools.product(ltn, lrn)).intersection(data)
		neighborhoods[t,r] = nbrs

	ntr = len(neighborhoods)
	share, srem = ntr // grpsize, ntr % grpsize
	start = rank * share + min(rank, srem)
	if rank < srem: share += 1

	# Sort desired arrival times by r-t, pull local share
	trpairs = set(sorted(neighborhoods)[start:start+share])
	# Only local neighborhoods are needed
	neighborhoods = { k: neighborhoods[k] for k in trpairs }

	# Build a map from elements to dependent neighborhoods
	nbrdeps = defaultdict(set)
	for nh, trpairs in neighborhoods.items():
		for tr in trpairs: nbrdeps[tr].add(nh)

	# Process waveforms in order, with transmit index varying most rapidly
	for tid, rid in sorted(nbrdeps):
		sig = data[tid, rid]

		if bandpass:
			# Remove DC bias to reduce Gibbs phenomenon
			sig.debias()
			# Bandpass and crop to original data window
			sig = sig.bandpass(**bandpass).window(sig.datawin)

		# Apply a window if desired
		if window: sig = applywindow(sig, (tid,rid), **window)

		# Yield the pair, waveform, and neighborhood
		yield (tid, rid), sig, nbrdeps[tid,rid]


def calcdelays(datafile, reffile, osamp=1, rank=0, grpsize=1, **kwargs):
	'''
	Given a datafile containing a habis.sigtools.WaveformMap, find arrival
	times using cross-correlation or IMER for waveforms returned by

	  wavegen(data, rank=rank, grpsize=grpsize, **exargs),

	where data is the WaveformMap encoded in datafile and exargs is a
	subset of kwargs as described below.

	For arrival times determined from cross-correlation, a reference
	waveform (as habis.sigtools.Waveform) is read from reffile. For IMER
	arrival times, reffile is ignored.

	The return value is a 2-tuple containing, first, a dictionary that maps
	a (t,r) transmit-receive index pair to delay in samples; and, second, a
	dictionary that maps stat groups to counts of waveforms that match the
	stats.

	Optional keyword arguments include:

	* flipref: A Boolean (default: False) that, when True, causes the
	  refrence waveform to be negated when read.

	* nsamp: Override data.nsamp. Useful mainly for bandpass filtering.

	* negcorr: A Boolean (default: False) passed to Waveform.delay as the
	  'negcorr' argument to consider negative cross-correlation.

	* signsquare: Square the waveform and reference amplitudes (multiplying
	  each signal by its absolute value to preserve signs) to better
	  emphasize peaks in the cross-correlation. The squaring is done right
	  after any bandpass filtering, so other parameters that are influence
	  by amplitude (e.g., minsnr, thresholds in peaks) should be altered to
	  account for the squared amplitudes.

	* minsnr: A sequence (mindb, noisewin) used to define the minimum
	  acceptable SNR in dB (mindb) by comparing the peak signal amplitude
	  to the minimum standard deviation over a sliding window of width
	  noisewin. SNR for each signal is calculated after application of an
	  optional window. Delays will not be calculated for signals fail to
	  exceed the minimum threshold.

	* denoise: If not None, a dictionary suitable for passing as keyword
	  arguments (**denoise) to Waveform.denoise to use CFAR rejection of
	  the Gabor spectrogram to isolate the signal. Denoising is done after
	  minimum-SNR rejection to avoid too many false matches with
	  very-low-noise signals.

	* peaks: A dictionary suitable for passing as keyword arguments
	  (**peaks) to the isolatepeak function, excluding the first three
	  arguments.

	  *** NOTE: peak windowing is done after overall windowing and after
	  possible exclusion by minsnr. ***

	* delaycache: A map from transmit-receive element pairs (t, r) to a
	  precomputed delay d. If a value exists for a given pair (t, r) in the
	  WaveformMap and the element map, the precomputed value will be used
	  in favor of explicit computation.

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

	* imer: A dictionary to provide all but the first argument of
	  getimertime. If this is provided, getimertime will be used instead of
	  (optional) peak isolation and cross-correlation to determine an
	  arrival time.

	* elements: If not None, an N-by-3 array or a map from element indices
	  to coordinates. If wavegen returns a neighborhood of more than one
	  transmit-receive pair for any arrival time, the element coordinates
	  will be used to find an optimal (in the least-squares sense) slowness
	  to predict arrivals observed in the neighborhood.

	  If an arrival-time measurement for the "key" pair in a measurement
	  neighborhood is available and average slowness imputed by this
	  arrival time falls within 1.5 IQR of the average slowness values for
	  all pairs in the neighborhood, or if the neighborhood consists of
	  only the key measurement pair, the arrival time for the "key" pair is
	  used without modification.

	  If the arrival time for a key pair is missing from the neighborhood,
	  or falls outside of 1.5 IQR, the arrival time for the key pair will
	  be the optimum slowness value for the neighborhood multiplied by the
	  propagation distance for the pair.

	  Element coordinates are required if wavegen returns neighborhoods of
	  more than one member.

	Any unspecified keyword arguments are passed to wavegen.
	'''
	# Read the data and reference
	data = WaveformMap.load(datafile)

	# Pull a copy of the IMER configuration, if it exists
	imer = dict(kwargs.pop('imer', ()))

	# Read the reference if IMER times are not desired
	if not imer:
		if reffile is None: raise ValueError('Must specify reffile or imer')
		ref = Waveform.fromfile(reffile)
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

	# Determine if an energy "leak" threshold is desired
	try:
		eleak = float(kwargs.pop('eleak'))
	except KeyError:
		eleak = None
	else:
		if not 0 <= eleak < 1:
			raise ValueError('Argument eleak must be in range [0, 1)')

	# Unpack minimum SNR requirements
	minsnr, noisewin = kwargs.pop('minsnr', (None, None))

	# Pull the optional peak search criteria
	peaks = dict(kwargs.pop('peaks', ()))

	# Pull the optional denoising criteria
	denoise = dict(kwargs.pop('denoise', ()))

	# Determine whether to allow negative correlations
	negcorr = kwargs.pop('negcorr', False)

	# Grab an optional delay cache
	delaycache = kwargs.pop('delaycache', { })

	# Grab an optional result queue
	queue = kwargs.pop('queue', None)

	# Element coordinates, if required
	elements = kwargs.pop('elements', None)

	# Pre-populate cached values
	result = { k: delaycache[k] for k in set(data).intersection(delaycache) }
	# Remove the cached waveforms from the set
	for k in result: data.pop(k, None)
	# Only keep a local portion of cached values
	result = { k: result[k] for k in sorted(result)[rank::grpsize] }

	wavestats = defaultdict(int)
	wavestats['cached'] = len(result)

	grpdelays = defaultdict(dict)

	# Process waveforms (possibly averages) as generated
	for key, sig, nbrs in wavegen(data, rank=rank, grpsize=grpsize, **kwargs):
		# Square the signal if desired
		if signsquare: sig = sig.signsquare()

		if minsnr is not None and noisewin is not None:
			if sig.snr(noisewin) < minsnr:
				wavestats['low-snr'] += 1
				continue

		if denoise: sig = sig.denoise(**denoise)

		# Calculate cumulative energy in unwindowed waveform
		if eleak: cenergy = np.cumsum(sig.data**2)

		if imer:
			# Compute IMER time
			try: dl = getimertime(sig, osamp=osamp, **imer)
			# Compute IMER and its mean
			except IndexError:
				wavestats['failed-IMER'] += 1
				continue
		else:
			if peaks:
				try: sig = isolatepeak(sig, key, **peaks)
				except ValueError:
					wavestats['missing-peak'] += 1
					continue

			# Compute and record the delay
			dl = sig.delay(ref, osamp=osamp, negcorr=negcorr)
			if negcorr:
				if dl[1] < 0: wavestats['negative-correlated'] += 1
				dl = dl[0]

		if eleak:
			# Evaluate leaked energy
			ssamp = int(dl) - sig.datawin.start - 1
			if not 0 <= ssamp < len(cenergy):
				wavestats['out-of-bounds'] += 1
				continue
			elif cenergy[ssamp] >= eleak * cenergy[-1]:
				wavestats['leaky'] += 1
				continue

		if len(nbrs) < 2:
			# If the element is its own neighborhood, just copy result
			if key in nbrs:
				wavestats['sole-valid'] += 1
				result[key] = dl
			else: wavestats['invalid-neighborhood'] += 1
		else:
			# Results will be optimized from groups of delays
			for nbr in nbrs: grpdelays[nbr][key] = dl

	if grpdelays and elements is None:
		raise TypeError('Cannot have neighborhoods when elements is None')

	for key, grp in grpdelays.items():
		if key[0] == key[1] or any(t == r for t, r in grp):
			raise ValueError('Backscatter neighborhoods not supported')

		pdist, slw = { }, { }
		try:
			# Find distances and speeds for neighborhoods
			for (t, r), dl in grp.items():
				v = norm(elements[t] - elements[r])
				pdist[t,r] = v
				slw[t,r] = dl / v
		except (KeyError, IndexError):
			# Either coordinates or a delay do not exist for
			wavestats['unknown-pair'] += 1
			continue

		# Eliminate outliers based on slowness; discard slowness values
		slw = set(stats.mask_outliers(slw))

		if key in slw:
			result[key] = grp[key]
			wavestats['valid-in-neighborhood'] += 1
		else: wavestats['outlier'] += 1

	try: queue.put((result, wavestats))
	except AttributeError: pass

	return result, stats


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

	def _throw(msg, e, sec=None):
		if not sec: sec = asec
		raise HabisConfigError.fromException(f'{msg} in [{sec}]', e)

	try:
		# Read all target input lists
		targets = sorted(k for k in config.options(asec) if k.startswith('target'))
		targetfiles = OrderedDict()
		for target in targets:
			targetfiles[target] = matchfiles(config.getlist(asec, target))
			if len(targetfiles[target]) < 1:
				raise HabisConfigError(f'Key {target} matches no inputs')
	except Exception as e:
		_throw('Configuration must specify at least one unique "target" key', e)

	try:
		efiles = config.getlist(asec, 'elements', default=None)
		if efiles:
			efiles = matchfiles(efiles)
			kwargs['elements'] = loadmatlist(efiles, nkeys=1)
	except Exception as e: _throw('Invalid optional elements', e)

	# Grab the reference file
	try: reffile = config.get(msec, 'reference', default=None)
	except Exception as e: _throw('Invalid optional reference', e, msec)

	# Grab the output file
	try: outfile = config.get(asec, 'outfile')
	except Exception as e: _throw('Configuration must specify outfile', e)

	try:
		# Grab the number of processes to use (optional)
		nproc = config.get('general', 'nproc', mapper=int,
				failfunc=process.preferred_process_count)
	except Exception as e: _throw('Invalid optional nproc', e, 'general')

	try:
		# Determine the sampling period and a global temporal offset
		dt = config.get(ssec, 'period', mapper=float)
		t0 = config.get(ssec, 'offset', mapper=float)
	except Exception as e:
		_throw('Configuration must specify period and offset', e, ssec)

	# Override the number of samples in WaveformMap
	try: kwargs['nsamp'] = config.get(ssec, 'nsamp', mapper=int)
	except HabisNoOptionError: pass
	except Exception as e: _throw('Invalid optional nsamp', e, ssec)

	# Determine the oversampling rate to use when cross-correlating
	try: osamp = config.get(ssec, 'osamp', mapper=int, default=1)
	except Exception as e: _throw('Invalid optional osamp', e, ssec)

	try:
		neighbors = config.get(asec, 'neighbors', default=None)
		if neighbors:
			kwargs['neighbors'] = loadkeymat(neighbors, dtype=int)
	except Exception as e: _throw('Invalid optional neighbors', e)

	# Determine the range of elements to use; default to all (as None)
	try: kwargs['minsnr'] = config.getlist(asec, 'minsnr', mapper=int)
	except HabisNoOptionError: pass
	except Exception as e: _throw('Invalid optional minsnr', e)

	# Determine a temporal window to apply before finding delays
	try: kwargs['window'] = config.get(asec, 'window')
	except HabisNoOptionError: pass
	except Exception as e: _throw('Invalid optional window', e)

	# Determine an energy leakage threshold
	try: kwargs['eleak'] = config.get(asec, 'eleak', mapper=float)
	except HabisNoOptionError: pass
	except Exception as e: _throw('Invalid optional eleak', e)

	# Determine a temporal window to apply before finding delays
	try: kwargs['bandpass'] = config.get(asec, 'bandpass')
	except HabisNoOptionError: pass
	except Exception as e: _throw('Invalid optional bandpass', e)

	# Determine peak-selection criteria
	try: kwargs['peaks'] = config.get(asec, 'peaks')
	except HabisNoOptionError: pass
	except Exception as e: _throw('Invalid optional peaks', e)

	# Determine IMER criteria
	try: kwargs['imer'] = config.get(asec, 'imer')
	except HabisNoOptionError: pass
	except Exception as e: _throw('Invalid optional imer', e)

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
		ftext = 'files' if len(datafiles) != 1 else 'file'
		print(f'Finding {dltype} delays for {target} ({len(datafiles)} {ftext})')

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
	except Exception as err:
		print('ERROR: could not load configuration '
				f'file {sys.argv[1]}: {err}', file=sys.stderr)
		usage(sys.argv[0])
		sys.exit(1)

	try: import pyfftw
	except ImportError: pass
	else:
		pyfftw.interfaces.cache.enable()
		pyfftw.interfaces.cache.set_keepalive_time(60.0)

	# Call the calculation engine
	try: atimesEngine(config)
	except Exception as e:
		print(f'Unable to complete arrival-time estimation: {e}', file=sys.stderr)
