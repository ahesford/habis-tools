#!/usr/bin/env python

import sys, itertools, ConfigParser, numpy as np
import multiprocessing, Queue

from pycwp import mio, process, cutil
from habis import sigtools, trilateration
from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.formats import WaveformSet


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def finddelays(datafile, reffile, osamp, nproc, delayfile=None):
	'''
	Compute the delay matrix for a habis.formats.WaveformSet stored in
	datafile. If delayfile is specified and can be read as a T x R matrix,
	where T and R are the "ntx" and "nrx" attributes of the WaveformSet,
	respectively, the delays are read from that file. Otherwise, the
	waveform set is read directly from datafile, a reference waveform is
	read from reffile in 1-D binary matrix form, and cross-correlation is
	used (with an oversampling rate of osamp) in the form of
	habis.sigtools.delay to determine the delay matrix.

	If delayfile is specified but computation is still required, the
	computed matrix will be saved to delayfile. Any existing content will
	be overwritten.

	Computations are done in parallel using nproc processes.
	'''
	# Grab the dimensions of the waveform matrix
	wset = WaveformSet.fromfile(datafile)
	t, r = wset.ntx, wset.nrx
	# No need to keep the WaveformSet around; close the file
	del wset

	try:
		# Try to read an existing delay file and check its size
		delays = np.loadtxt(delayfile)
		if delays.shape != (t, r):
			raise ValueError('Delay file has wrong shape; recomputing')
		return delays
	except (ValueError, IOError):
		# ValueError if format is inappropriate or delayfile is None
		# IOError if delayfile does not point to an existent file
		pass

	# Create a result queue and a matrix to store the accumulated results
	queue = multiprocessing.Queue(nproc)
	delays = np.zeros((t, r), dtype=np.float32)

	# Spawn the desired processes to perform the cross-correlation
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Pick a useful process name
			procname = process.procname(i)
			args = (datafile, reffile, osamp, queue, i, nproc)
			pool.addtask(target=calcdelays, name=procname, args=args)

		pool.start()

		# Wait for all processes to respond
		responses = 0
		while responses < nproc:
			try:
				results = queue.get(timeout=0.1)
				for idx, delay in results:
					row, col = np.unravel_index(idx, (t, r), 'C')
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


def calcdelays(datafile, reffile, osamp, queue=None, start=0, stride=1):
	'''
	Given a datafile containing a habis.formats.WaveformSet, perform
	cross-correlation on every stride-th received waveform, starting at
	index start, to identify the delay of the received waveform relative to
	the reference. The waveforms are oversampled by the factor osamp when
	performing the cross-correlation. The start index and stride span an
	array of T x R waveforms flattened in row-major order.

	The reference file must be a 1-dimensional matrix as a float32.
	Waveforms in the datafile will be cast to float32.

	The return value is a list of 2-tuples, wherein the first element is a
	flattened waveform index and the second element is the corresponding
	delay in samples. If queue is not None, the result list is also placed
	in the queue using queue.put().
	'''
	# Read the data and reference files
	data = WaveformSet.fromfile(datafile)
	# Force the proper data type and shape for the reference
	ref = mio.readbmat(reffile, dim=1, dtype=np.float32)

	# Pull the relevant waveform set dimensions
	t, r, ns = data.ntx, data.nrx, data.nsamp

	if ref.shape[0] != ns:
		raise TypeError('Number of samples in data and reference waveforms must agree')

	# Compute the strided results
	result = []
	for idx in range(start, t * r, stride):
		# Find the transmit and receive indices
		tid, rid = np.unravel_index(idx, (t, r), 'C')
		# Pull the waveform in the right format and shape (ignore header)
		sig = data.getrecord(rid, tid, (0, ns), np.float32)[1]
		# Comput the delay and append to the result
		result.append((idx, sigtools.delay(sig, ref, osamp)))

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
		datafiles = []
		delayfiles = {}
		# Loop through all [atimes] records looking for datafiles
		for key, value in config.items('atimes'):
			if key.startswith('datafile'):
				# Add the datafile to the list
				datafiles.append(value)
				# A matching delay file can be specified by
				# swapping 'datafile' with 'delayfile' in key
				delaykey = key.replace('datafile', 'delayfile', 1)
				# The delayfile is optional, but if found, the
				# dictionary is keyed on the data file name
				try: delayfiles[value] = config.get('atimes', delaykey)
				except ConfigParser.NoOptionError: pass
				
	except ConfigParser.Error:
		raise HabisConfigError('Configuration must specify an [atimes] section')

	if len(datafiles) < 1:
		raise HabisConfigError('Configuration must specify at least one datafile in [atimes]')

	try:
		# Grab the reference and output files
		reffile = config.get('atimes', 'reffile')
		outfile = config.get('atimes', 'outfile')
	except ConfigParser.Error:
		raise HabisConfigError('Configuration must specify reference and output files in [atimes]')

	try:
		# Grab the oversampling rate
		osamp = config.getint('atimes', 'osamp')
	except (ConfigParser.Error, ValueError):
		raise HabisConfigError('Configuration must specify oversampling rate as integer in [atimes]')

	try:
		# Grab the number of processes to use (optional)
		nproc = config.getint('general', 'nproc')
	except ConfigParser.NoOptionError:
		nproc = process.preferred_process_count()
	except:
		raise HabisConfigError('Invalid specification of process count in [general]')

	try:
		# Determine the number of samples and offset, in microsec
		dt = config.getfloat('sampling', 'period')
		t0 = config.getfloat('sampling', 'offset')
	except:
		raise HabisConfigError('Configuration must specify float sampling period and temporal offset in [sampling]')

	symmetrize = config.getbooldefault('atimes', 'symmetrize', False)
	usediag = config.getbooldefault('atimes', 'usediag', False)
	maskoutliers = config.getbooldefault('atimes', 'maskoutliers', False)

	# Determine the shape of the first multipath data
	t, r, ns = mio.getmattype(datafiles[0], dim=3, dtype=np.float32)[0]

	# Check that all subsequent data files have the same shape
	for datafile in datafiles[1:]:
		tp, rp, nsp = mio.getmattype(datafile, dim=3, dtype=np.float32)[0]
		if (tp, rp, nsp) != (t, r, ns):
			raise TypeError('All input waveform data must have same shape')

	# Store results for all data files in this list
	times = []

	for datafile in datafiles:
		# Try to read or compute the delay matrix for this data set
		delayfile = delayfiles.get(datafile, None)
		print 'Finding delays for data set', datafile
		delays = finddelays(datafile, reffile, osamp, nproc, delayfile)

		# Convert delays to arrival times
		delays = delays * dt + t0

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
	config = HabisConfigParser()
	try:
		config.readfp(open(sys.argv[1]))
	except:
		print >> sys.stderr, 'ERROR: could not load configuration file %s' % sys.argv[1]
		usage(sys.argv[0])
		sys.exit(1)

	# Call the calculation engine
	atimesEngine(config)
