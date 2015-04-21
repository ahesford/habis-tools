#!/usr/bin/env python

import os, sys, itertools, ConfigParser, numpy as np
import ctypes, multiprocessing
import socket

from operator import mul

from pycwp import mio, process
from habis import sigtools, trilateration

# Define a new ConfigurationError exception
class ConfigurationError(Exception): pass

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def realarray(shape):
	'''
	Create a multiprocessing.Array shared instance to act as a backing
	store for a Numpy array of float32 values with the specified shape.
	
	Returns the tuple (base, narray), where base is the multiprocess.Array
	instance and narray is the Numpy structure atop the backer. The Numpy
	array is stored in row-major (C) order.
	'''
	base = multiprocessing.Array(ctypes.c_float, reduce(mul, shape))
	narray = np.frombuffer(base.get_obj(), dtype=np.float32).reshape(shape)

	return base, narray


def getdelays(datafile, reffile, osamp, output=None, start=0, stride=1):
	'''
	Given a datafile that contains a T x R x Ns matrix of Ns-sample
	waveforms transmitted by T elements and received by R elements, and a
	corresponding Ns-sample reference waveform stored in reffile, perform
	cross-correlation on every received waveform for every stride transmit
	waveforms, starting at index start, to identify the delay of the
	received waveform relative to the reference. The waveforms are
	oversampled by the factor osamp when performing the cross-correlation.

	The data file must contain a 3-dimensional matrix, while the reference
	must be a 1-dimensional matrix. Both files must contain 32-bit
	floating-point values.

	The computed delays populate rows of the provided matrix mat. If mat is
	None, a new Numpy array is created and filled with zeros wherever
	stride skips values. The output array is always returned.
	'''
	# Read the data and reference files; force proper shapes and dtypes
	data = mio.readbmat(datafile, dim=3, dtype=np.float32)
	ref = mio.readbmat(reffile, dim=1, dtype=np.float32)

	if data.shape[-1] != ref.shape[0]:
		raise TypeError('Number of samples in data and reference waveforms must agree')

	# If the output is not provided, create it
	if output is None:
		output = np.zeros(data.shape[:-1], dtype=np.float32)

	if data.shape[:-1] != output.shape:
		raise TypeError('Shape of output must match first two dimensions of data')

	# Process the results for the appropriate rows
	for row in range(start, data.shape[0], stride):
		output[row] = [sigtools.delay(sig, ref, osamp) for sig in data[row]]

	return output


def atimesEngine(config):
	'''
	Use habis.trilateration.ArrivalTimeFinder to determine a set of
	round-trip arrival times from a set of one-to-many multipath arrival
	times. Multipath arrival times are computed as the maximum of
	cross-correlation with a reference pulse, plus some constant offset.
	'''
	try:
		# Grab the data and reference files
		datafile = config.get('atimes', 'datafile')
		reffile = config.get('atimes', 'reffile')
		outfile = config.get('atimes', 'outfile')
	except ConfigParser.Error:
		raise ConfigurationError('Configuration must specify data, reference, and output files in [atimes]')

	try:
		# Grab the oversampling rate
		osamp = int(config.get('atimes', 'osamp'))
	except (ConfigParser.Error, ValueError):
		raise ConfigurationError('Configuration must specify oversampling rate as integer in [atimes]')

	try:
		# Grab the number of processes to use (optional)
		nproc = int(config.get('general', 'nproc'))
	except ConfigParser.Error:
		nproc = process.preferred_process_count()
	except:
		raise ConfigurationError('Invalid specification of process count in [general]')

	try:
		# Determine the number of samples and offset, in microsec
		dt = float(config.get('sampling', 'period'))
		t0 = float(config.get('sampling', 'offset'))
	except:
		raise ConfigurationError('Configuration must specify float sampling period and temporal offset in [sampling]')

	# Determine the shape of the multipath data
	shape = mio.getmattype(datafile, dim=3, dtype=np.float32)[0]

	# Create the shared memory that will store the delays
	dlbase, delays = realarray(shape[:-1])

	# Spawn the desired processes to perform the cross-correlation
	with process.ProcessPool() as pool:
		hostname = socket.gethostname()
		execname = os.path.basename(sys.argv[0])

		for i in range(nproc):
			# Pick a useful hostname
			procname = '{:s}-{:s}-Rank{:d}'.format(hostname, execname, i)
			args = (datafile, reffile, osamp, delays, i, nproc)
			pool.addtask(target=getdelays, name=procname, args=args)

		pool.start()
		pool.wait()

	# Compute the actual signal arrival times
	delays = delays * dt + t0

	# Prepare the arrival-time finder
	atf = trilateration.ArrivalTimeFinder(delays)
	times = atf.lsmr()

	# Save the output as a text file
	np.savetxt(outfile, times, fmt='%16.8f')


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
