#!/usr/bin/env python

import sys, itertools, ConfigParser, numpy as np
import multiprocessing, Queue

from pycwp import process, cutil
from habis import trilateration
from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.formats import WaveformSet
from habis.sigtools import Waveform


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def finddelays(datafile, reffile, osamp, nproc,
		window=None, peaks=None, elements=None, delayfile=None):
	'''
	Compute the delay matrix for a habis.formats.WaveformSet stored in
	datafile. If delayfile is specified and can be read as a T x R matrix,
	where T and R are the "ntx" and "nrx" attributes of the WaveformSet,
	respectively, the delays are read from that file. Otherwise, the
	waveform set is read directly from datafile, a reference waveform is
	read from reffile in 1-D binary matrix form, and cross-correlation is
	used (with an oversampling rate of osamp) in the form of
	habis.sigtools.Waveform.delay to determine the delay matrix.

	If elements is not None, it should be a list of element indices, and
	entry (i, j) in the delay matrix will correspond to the delay of the
	waveform for transmit index elements[i] and receive index elements[j].

	If window is not None, it specifies a window passed to calcdelays that
	is used to window the waveforms before finding delays.

	If peaks is not None, it specifies a dictionary of kwargs to be passed
	to Waveform.envpeaks() for every waveform. The waveform will be
	windowed around the first peak identified by this method. The width of
	the window extends to the width of the identified peak in either
	direction of the peak. No tails are used. Peak windowing is done after
	overall signal windowing.

	If delayfile is specified but computation is still required, the
	computed matrix will be saved to delayfile. Any existing content will
	be overwritten.

	Computations are done in parallel using nproc processes.
	'''
	# Grab the dimensions of the waveform matrix
	wset = WaveformSet.fromfile(datafile)
	# Determine the shape of the delay matrix
	try: wshape = len(elements), len(elements)
	except TypeError: wshape = wset.ntx, wset.nrx
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

	# Spawn the desired processes to perform the cross-correlation
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Pick a useful process name
			procname = process.procname(i)
			args = (datafile, reffile, osamp, window, 
					peaks, elements, queue, i, nproc)
			pool.addtask(target=calcdelays, name=procname, args=args)

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


def calcdelays(datafile, reffile, osamp, window=None,
		peaks=None, elements=None, queue=None, start=0, stride=1):
	'''
	Given a datafile containing a habis.formats.WaveformSet, perform
	cross-correlation on every stride-th received waveform, starting at
	index start, to identify the delay of the received waveform relative to
	the reference. The waveforms are oversampled by the factor osamp when
	performing the cross-correlation. The start index and stride span an
	array of T x R waveforms flattened in row-major order.

	The reference file must be a 1-dimensional matrix as a float32.
	Waveforms in the datafile will be cast to float32.

	If window is not None, it should be a tuple of ints of the form
	(start, length, [tailwidth]) that specifies a window to be applied to
	each waveform before finding delays. If tailwidth is provided, a Hann
	window of length (2 * tailwidth) is passed as the tails argument to
	habis.sigtools.Waveform.window().

	If peaks is not None, it specifies a dictionary of kwargs to be passed
	to Waveform.envpeaks() for every waveform. The waveform will be
	windowed around the first peak identified by this method. The width of
	the window extends to the width of the identified peak in either
	direction of the peak. No tails are used. Peak windowing is done after
	overall signal windowing.

	If elements is not None, it should be a list of element indices, and
	entry (i, j) in the delay matrix will correspond to the delay of the
	waveform for transmit index elements[i] and receive index elements[j].

	The return value is a list of 2-tuples, wherein the first element is a
	flattened waveform index and the second element is the corresponding
	delay in samples. If queue is not None, the result list is also placed
	in the queue using queue.put().
	'''
	# Read the data and reference files
	data = WaveformSet.fromfile(datafile)
	# Read the reference waveform
	ref = Waveform.fromfile(reffile)

	# Determine the shape of the delay matrix
	try: t, r = len(elements), len(elements)
	except TypeError: t, r = data.ntx, data.nrx

	# Pull the tails of an optional window
	try: tails = np.hanning(2 * window[2])
	except (TypeError, IndexError): tails = None

	# Compute the strided results
	result = []
	for idx in range(start, t * r, stride):
		# Find the transmit and receive indices
		i, j = np.unravel_index(idx, (t, r), 'C')
		tid, rid = elements[i], elements[j]
		# Pull the waveform as float32
		sig = data.getwaveform(rid, tid, dtype=np.float32)
		if window: sig = sig.window(window[:2], tails=tails)
		if peaks:
			# Find the earliest peak in the signal
			ctr, _, width, _ = min(sig.envpeaks(**peaks), key=lambda pk: pk[0])
			sig = sig.window((ctr - width, 2 * width))
		# Compute the delay (may be negative)
		result.append((idx, sig.delay(ref, osamp,)))

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
	try:
		# Read the list of data files
		datafiles = config.getlist('atimes', 'datafile')
		if len(datafiles) < 1:
			raise ConfigParser.Error('Fall-through to exception handler')
	except:
		raise HabisConfigError('Configuration must specify at least one datafile')

	# Read an optional list of delay files
	# Default delay files are empty (no files)
	try:
		delayfiles = config.getlist('atimes', 'delayfile', 
				failfunc=lambda: [''] * len(datafiles))
	except:
		raise HabisConfigError('Invalid specification of delayfile in [atimes]')

	# Make sure the delayfile and datafile counts match
	if len(datafiles) != len(delayfiles):
		raise HabisConfigError('Number of delay files must match number of data files')

	try:
		# Grab the reference and output files
		reffile = config.get('atimes', 'reffile')
		outfile = config.get('atimes', 'outfile')
	except:
		raise HabisConfigError('Configuration must specify reference and output files in [atimes]')

	try:
		# Grab the oversampling rate
		osamp = config.getint('atimes', 'osamp')
	except:
		raise HabisConfigError('Configuration must specify oversampling rate as integer in [atimes]')

	try:
		# Grab the number of processes to use (optional)
		nproc = config.getint('general', 'nproc',
				failfunc=process.preferred_process_count)
	except:
		raise HabisConfigError('Invalid specification of process count in [general]')

	try:
		# Determine the number of samples and offset, in microsec
		dt = config.getfloat('sampling', 'period')
		t0 = config.getfloat('sampling', 'offset')
	except:
		raise HabisConfigError('Configuration must specify float sampling period and temporal offset in [sampling]')

	try:
		# Determine the range of elements to use; default to all (as None)
		elements = config.getrange('atimes', 'elements', failfunc=lambda: None)
	except:
		raise HabisConfigError('Invalid specification of optional element indices in [atimes]')

	try:
		# Determine a temporal window to apply before finding delays
		window = config.getlist('atimes', 'window',
				mapper=int, failfunc=lambda: None)
		if window and (len(window) < 2 or len(window) > 3):
			raise ValueError('Fall-through to exception handler')
	except:
		raise HabisConfigError('Invalid specification of optional temporal window in [atimes]')

	try:
		# Determine peak-selection criteria
		peaks = config.getlist('atimes', 'peak', failfunc=lambda: None)
		if peaks:
			if len(peaks) < 2 or len(peaks) > 4:
				raise ValueError('Fall-through to exception handler')
			if peaks[0].lower() != 'first':
				raise ValueError('Fall-through to exception handler')
			# Build the argument list
			peakargs = {'minwidth': float(peaks[1]) }
			if len(peaks) > 2:
				peakargs['minprom'] = float(peaks[2])
			if len(peaks) > 3:
				peakargs['prommode'] = peaks[3]
			peaks = peakargs
	except:
		raise HabisConfigError('Invalid specification of optional peak-selection criteria')

	symmetrize = config.getboolean('atimes', 'symmetrize', failfunc=lambda: False)
	usediag = config.getboolean('atimes', 'usediag', failfunc=lambda: False)
	maskoutliers = config.getboolean('atimes', 'maskoutliers', failfunc=lambda: False)

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

	for datafile, delayfile in zip(datafiles, delayfiles):
		# If an empty name is specified, use no delayfile
		if len(delayfile) == 0: delayfile = None
		print 'Finding delays for data set', datafile
		delays = finddelays(datafile, reffile, osamp,
				nproc, window, peaks, elements, delayfile)

		# Convert delays to arrival times
		delays = delays * dt + t0
		# Negative arrival times should be artifacts of delay analysis
		# Try wrapping the delays into a positive window
		delays[delays < 0] += wnsamp * dt
		if np.any(delays < 0):
			raise ValueError('Negative delays exist, but are non-phyiscal')

		if usediag:
			# Skip optimization in favor of matrix diagonal
			times.append(np.diag(delays))
			continue

		if maskoutliers: delays = cutil.mask_outliers(delays)
		if symmetrize: delays = 0.5 * (delays + delays.T)

		# Prepare the arrival-time finder
		atf = trilateration.ArrivalTimeFinder(delays)
		# Compute the times for this data file and add to the result list
		times.append(atf.lsmr())

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
