#!/usr/bin/env python

import os, sys, itertools, ConfigParser, numpy as np
import multiprocessing, Queue
import socket
import json

from operator import mul

from pycwp import mio, process
from habis import sigtools, trilateration

# Define a new ConfigurationError exception
class ConfigurationError(Exception): pass

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def getdelays(datafile, reffile, osamp, queue=None, start=0, stride=1):
	'''
	Given a datafile that contains a T x R x Ns matrix of Ns-sample
	waveforms transmitted by T elements and received by R elements, and a
	corresponding Ns-sample reference waveform stored in reffile, perform
	cross-correlation on every stride-th received waveform, starting at
	index start, to identify the delay of the received waveform relative to
	the reference. The waveforms are oversampled by the factor osamp when
	performing the cross-correlation. The start index and stride span array
	of T x R waveforms flattened in row-major order.

	The data file must contain a 3-dimensional matrix, while the reference
	must be a 1-dimensional matrix. Both files must contain 32-bit
	floating-point values.

	The return value is a list of 2-tuples, wherein the first element is
	a flattened waveform index and the second element is the corresponding
	delay in samples. If queue is not None, the result list is also placed
	in the queue using queue.put().
	'''
	# Read the data and reference files; force proper shapes and dtypes
	data = mio.readbmat(datafile, dim=3, dtype=np.float32)
	ref = mio.readbmat(reffile, dim=1, dtype=np.float32)

	t, r, ns = data.shape

	if ref.shape[0] != ns:
		raise TypeError('Number of samples in data and reference waveforms must agree')


	# Compute the strided results
	result = []
	for idx in range(start, t * r, stride):
		row, col = np.unravel_index(idx, (t, r), 'C')
		result.append((idx, sigtools.delay(data[row,col], ref, osamp)))

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
		# Grab the reference and output files
		reffile = config.get('atimes', 'reffile')
		outfile = config.get('atimes', 'outfile')
	except ConfigParser.Error:
		raise ConfigurationError('Configuration must specify reference and output files in [atimes]')

	try:
		# Try to grab the list of data files
		datalist = config.get('atimes', 'data')
	except ConfigParser.Error:
		raise ConfigurationError('Configuration must specify input data files in [atimes]')

	try:
		# Assume the list of data files is a valid JSON object
		datafiles = json.loads(datalist)
	except ValueError:
		# If JSON decoding fails, treat the list as whitespace-delimited strings
		datafiles = datalist.split()

	try:
		# Grab the oversampling rate
		osamp = int(config.get('atimes', 'osamp'))
	except (ConfigParser.Error, ValueError):
		raise ConfigurationError('Configuration must specify oversampling rate as integer in [atimes]')

	try:
		# Grab the number of processes to use (optional)
		nproc = int(config.get('general', 'nproc'))
	except ConfigParser.NoOptionError:
		nproc = process.preferred_process_count()
	except:
		raise ConfigurationError('Invalid specification of process count in [general]')

	try:
		# Determine the number of samples and offset, in microsec
		dt = float(config.get('sampling', 'period'))
		t0 = float(config.get('sampling', 'offset'))
	except:
		raise ConfigurationError('Configuration must specify float sampling period and temporal offset in [sampling]')

	try:
		symmetrize = config.getboolean('atimes', 'symmetrize')
	except ConfigParser.NoOptionError:
		symmetrize = False
	except:
		raise ConfigureError('Invalid specification of symmetrize in [atimes]')

	# Determine the shape of the first multipath data
	t, r, ns = mio.getmattype(datafiles[0], dim=3, dtype=np.float32)[0]

	# Check that all subsequent data files have the same shape
	for datafile in datafiles[1:]:
		tp, rp, nsp = mio.getmattype(datafile, dim=3, dtype=np.float32)[0]
		if (tp, rp, nsp) != (t, r, ns):
			raise TypeError('All input waveform data must have same shape')

	# Store results for all data files in this list
	times = []

	# Grab the hostname and executable name for pretty process naming
	hostname = socket.gethostname()
	execname = os.path.basename(sys.argv[0])

	for datafile in datafiles:
		# Create a result queue and a list to store the accumulated results
		queue = multiprocessing.Queue(nproc)
		delays = np.zeros((t, r), dtype=np.float32)

		# Spawn the desired processes to perform the cross-correlation
		with process.ProcessPool() as pool:

			for i in range(nproc):
				# Pick a useful hostname
				procname = '{:s}-{:s}-Rank{:d}'.format(hostname, execname, i)
				args = (datafile, reffile, osamp, queue, i, nproc)
				pool.addtask(target=getdelays, name=procname, args=args)

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

		# Compute the actual signal arrival times
		delays = delays * dt + t0
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
	config = ConfigParser.SafeConfigParser()
	if len(config.read(sys.argv[1])) == 0:
		print >> sys.stderr, 'ERROR: configuration file %s does not exist' % sys.argv[1]
		usage(sys.argv[0])
		sys.exit(1)

	# Call the calculation engine
	atimesEngine(config)
