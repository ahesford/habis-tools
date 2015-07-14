#!/usr/bin/env python

import sys, itertools, numpy as np
import multiprocessing, Queue

from numpy import ma

from itertools import izip

from pycwp import process, cutil
from habis import trilateration
from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.formats import WaveformSet
from habis.sigtools import Waveform


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def finddelays(datafile, delayfile=None, nproc=1, *args, **kwargs):
	'''
	Compute the delay matrix for a habis.formats.WaveformSet stored in
	datafile. If delayfile is specified and can be read as a T x R matrix,
	where T and R are the "ntx" and "nrx" attributes of the WaveformSet,
	respectively, the delays are read from that file. Otherwise, the
	waveform set is read directly from datafile and delays are determined
	through delay analysis.
	
	Delay analysis for individual waveforms is farmed out to calcdelays()
	among nproc processes. The datafile argument, along with *args and
	**kwargs, are passed to calcdelays on each participating process. This
	function explicitly sets the "queue", "start", and "stride" arguments
	of calcdelays, so *args and **kwargs should not contain these values.

	If delayfile is specified but computation is still required, the
	computed matrix will be saved to delayfile. Any existing content will
	be overwritten.
	'''
	# Grab the dimensions of the waveform matrix
	wset = WaveformSet.fromfile(datafile)
	# Determine the shape of the delay matrix
	try: tlen = len(kwargs['txelts'])
	except (KeyError, TypeError): tlen = wset.ntx
	try: rlen = len(kwargs['rxelts'])
	except (KeyError, TypeError): rlen = wset.nrx
	wshape = (tlen, rlen)
	# No need to keep the WaveformSet around; kill the memmap
	del wset

	try:
		# Try to read an existing delay file and check its size
		delays = np.loadtxt(delayfile)
		if delays.shape != wshape:
			raise ValueError('Delay file has wrong shape; recomputing')
		return delays
	except (ValueError, IOError):
		# ValueError if format is inappropriate or delayfile is None
		# IOError if delayfile does not point to an existent file
		pass

	# Create a result queue and a matrix to store the accumulated results
	queue = multiprocessing.Queue(nproc)
	delays = np.zeros(wshape, dtype=np.float32)

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
			try:
				results = queue.get(timeout=0.1)
				for idx, delay in results:
					row, col = np.unravel_index(idx, wshape, 'C')
					delays[row,col] = delay
				responses += 1
			except Queue.Empty: pass

		pool.wait()

	try:
		# Save the computed delays, if desired
		np.savetxt(delayfile, delays, fmt='%16.8f')
	except (ValueError, IOError):
		pass

	return delays


def calcdelays(datafile, reffile, osamp, start=0, stride=1, **kwargs):
	'''
	Given a datafile containing a habis.formats.WaveformSet, perform
	cross-correlation on every stride-th waveform, starting at index start,
	to identify the delay of the received waveform relative to a reference
	as the output of datafile[i,j].delay(reference, osamp=osamp). The index
	(i,j) of the WaveformSet is determined by un-flattening the strided
	index, k, into a 2-D index (j,i) into the T x R delay matrix in
	row-major order.

	The reference waveform is read from reffile using
	habis.sigtools.Waveform.fromfile.

	The return value is a list of 2-tuples, wherein the first element is a
	flattened waveform index and the second element is the corresponding
	delay in samples.

	The optional kwargs are parsed for the following keys:

	* window: If not None, should be a (start, length, [tailwidth]) tuple
	  of ints that specifies a temporal window applied to each waveform
	  before delay analysis. Optional tailwidth specifies the half-width of
	  a Hann window passed as the tails argument to Waveform.window.

	* peaks: If not None, should be a dictionary of kwargs to be passed to
	  Waveform.envpeaks for every waveform. Additionally, a 'nearmap' key
	  must be included to specify a list of expected round-trip delays for
	  each element. The waveform (i,j) will be windowed about the peak
	  closest to delay 0.5 * (nearmap[i] + nearmap[j]) to a width twice the
	  peak width, with no tails. Note: peak windowing is done after overall
	  windowing.

	* rxelts and txelts: If not None, should be a lists of element indices
	  such that entry (i,j) in the delay matrix corresponds to the waveform
	  datafile[rxelts[j],txelts[i]]. When rxelts or txelts are None or
	  unspecified, they are populated by sorted(datafile.rxidx) and
	  sorted(datafile.txidx), respectively.

	* compenv: If True, delay analysis will proceed on signal and reference
	  envelopes. If false, delay analysis uses the original signals.

	* queue: If not none, the return list is passed as an argument to
	  queue.put().
	'''
	# Read the data and reference
	data = WaveformSet.fromfile(datafile)
	ref = Waveform.fromfile(reffile)

	# Determine the elements to include in the delay matrix
	rxelts = kwargs.get('rxelts', None)
	if rxelts is None: rxelts = sorted(data.rxidx)
	txelts = kwargs.get('txelts', None)
	if txelts is None: txelts = sorted(data.txidx)

	# Use envelopes for delay analysis if desired
	compenv = kwargs.get('compenv', False)
	if compenv: ref = ref.envelope()

	t, r = len(txelts), len(rxelts)

	# Pull the window and compute optional tails
	window = kwargs.get('window', None)
	try: tails = np.hanning(2 * window[2])
	except (TypeError, IndexError): tails = None

	# Pull the optional peak search criteria
	peaks = kwargs.get('peaks', None)
	try:
		nearmap = peaks.get('nearmap', None)
		peaks = dict(kp for kp in peaks.iteritems() if kp[0] != 'nearmap')
	except AttributeError: nearmap = None

	if peaks and nearmap is None:
		raise KeyError('kwarg "peaks" must be a dictionary with a "nearmap" key')

	# Compute the strided results
	result = []
	for idx in range(start, t * r, stride):
		# Find the transmit and receive indices
		i, j = np.unravel_index(idx, (t, r), 'C')
		tid, rid = txelts[i], rxelts[j]
		# Pull the waveform as float32
		sig = data.getwaveform(rid, tid, dtype=np.float32)
		if window: sig = sig.window(window[:2], tails=tails)
		if peaks:
			# Find any signal peaks
			pks = sig.envpeaks(**peaks)
			if len(pks) > 1:
				# Choose the peak closest to nearmap prediction
				exd = 0.5 * (nearmap[tid] + nearmap[rid])
				ctr, _, width, _ = min(pks, key=lambda pk: abs(pk[0] - exd))
				sig = sig.window((ctr - width, 2 * width))
		if compenv: sig = sig.envelope()
		# Compute the delay (may be negative)
		result.append((idx, sig.delay(ref, osamp)))

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
		# Read the list of data files
		datafiles = config.getlist(asec, 'waveset')
		if len(datafiles) < 1:
			err = 'Key waveset present, but specifies no files'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify waveset in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	# Read an optional list of delay files
	# Default delay files are empty (no files)
	try:
		delayfiles = config.getlist(asec, 'delayfile',
				failfunc=lambda: [''] * len(datafiles))
	except Exception as e:
		err = 'Invalid specification of optional delayfile in [%s]' % asec
		raise HabisConfigError.fromException(err, e)

	# Make sure the delayfile and datafile counts match
	if len(datafiles) != len(delayfiles):
		err = 'Number of delay files must match number of data files'
		raise HabisConfigError(err)

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
		# Determine the number of samples and time offset
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

	symmetrize = config.getboolean(asec, 'symmetrize', failfunc=lambda: False)
	usediag = config.getboolean(asec, 'usediag', failfunc=lambda: False)
	maskoutliers = config.getboolean(asec, 'maskoutliers', failfunc=lambda: False)
	compenv = config.getboolean(asec, 'compenv', failfunc=lambda: False)

	# Determine the shape of the multipath data
	wset = WaveformSet.fromfile(datafiles[0])
	wshape = wset.ntx, wset.nrx
	wnsamp = wset.nsamp

	# Check that all subsequent data files have the same shape
	for datafile in datafiles[1:]:
		wset = WaveformSet.fromfile(datafile)
		if (wset.ntx, wset.nrx) != wshape:
			raise TypeError('All input waveform data must have same shape')

	# Allow the memmap to be closed
	del wset

	# Store results for all data files in this list
	times = []

	try:
		# If a delay guess was specified, read the delay matrix
		guesses = np.loadtxt(peaks['nearfile'])
		# Convert from times to samples
		guesses = (guesses - t0) / dt
		# Remove the "nearfile" key
		del peaks['nearfile']
	except (KeyError, TypeError, IOError, AttributeError):
		guesses = None

	for i, (dfile, dlayfile) in enumerate(izip(datafiles, delayfiles)):
		# If an empty name is specified, use no delayfile
		if len(dlayfile) == 0: dlayfile = None
		print 'Finding delays for data set', dfile

		# Set the peaks nearmap to the appropriate guess column
		try: peaks['nearmap'] = guesses[:,i]
		except TypeError: pass

		delays = finddelays(dfile, dlayfile, nproc, reffile, osamp,
				compenv=compenv, window=window, 
				peaks=peaks, rxelts=rxelts, txelts=txelts)

		# Convert delays to arrival times
		delays = delays * dt + t0
		# Negative arrival times should be artifacts of delay analysis
		# Try wrapping the delays into a positive window
		delays[delays < 0] += wnsamp * dt
		if np.any(delays < 0):
			raise ValueError('Negative delays exist, but are non-phyiscal')

		if maskoutliers: delays = cutil.mask_outliers(delays)
		if symmetrize: delays = 0.5 * (delays + delays.T)

		# Prepare the arrival-time finder
		atf = trilateration.ArrivalTimeFinder(delays)
		if not usediag:
			# Compute the optimized times for this data file
			optimes = atf.lsmr()
		elif maskoutliers:
			# Preserve any masked entries
			optimes = ma.diag(delays)
			# Replace masked diagonals with optimized times
			mask = ma.getmaskarray(optimes)
			if np.any(mask):
				rtimes = atf.lsmr()
				optimes = np.where(mask, rtimes, optimes)
		else:
			# Just take the diagonal if values are not masked
			optimes = np.diag(delays)

		times.append(optimes)

	# Save the output as a text file
	# Each data file gets a column
	np.savetxt(outfile, np.transpose(times), fmt='%16.8f')


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
