#!/usr/bin/env python

import sys, itertools, numpy as np
import multiprocessing, Queue

from pycwp import process
from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.formats import WaveformSet
from habis.sigtools import Waveform


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def copyhdr(f, wset, dtype=None, ver=(1,0)):
	'''
	Copy the file header and list of transmit channels from the
	habis.formats.WaveformSet wset to f, a file-like object or the name of
	a target file. If f is a file-like objec, the data is written at the
	current file position. If f is a name, the named file is truncated.

	If dtype is provided, it specifies a datatype to override the waveform
	record datatype in wset.

	The header format version can be specified as ver=(major, minor).
	'''
	# Open a named file, if necessary
	if isinstance(f, basestring): f = open(f, 'wb')

	if dtype is None: dtype = wset.dtype

	# Copy the relevant header fields and write the header
	nchans = (wset.nrx, wset.ntx)
	hdr = wset.packfilehdr(dtype, nchans, wset.nsamp, wset.f2c, wset.txidx, ver)
	f.write(hdr)


def wavefilt(infile, rxchans, filt, outfile, lock=None, nsamp=None):
	'''
	For the habis.formats.WaveformSet object stored in infile, successively
	filter all waveforms received by the channels specified in the sequence
	rxchans use a bandpass filter (from habis.sigtools.Waveform.bandpass).
	Append the filtered waveforms to the specified file named by outfile,
	which should exist and already contain a compatible WaveformSet file
	header. All output waveforms will be of type float32. If lock is
	provided, it will be acquired and released by calling lock.acquire()
	and lock.release(), respectively, immediately prior to and following
	the append.

	The filter is defined by the tuple filt = (start, end, [tailwidth]).
	The bandwidth start and end parameters are specified in units of R2C
	DFT bin widths, where the total DFT length is governed by the "nsamp"
	parameter of the WaveformSet serialized in infile. These parameters are
	passed directly to the corresponding arguments of bandpass(). The
	optional tailwidth, if provided, should be a positive integer
	specifying the half-width of a Hann window passed as the tails
	argument to bandpass(). The Hann window is used to roll off the edges
	of the bandpass filter inside the (start, end) interval.

	If nsamp is None, the input waveforms are truncated to nsamp samples.
	The output will also be truncated.

	** NOTE **
	If this function is used in a multiprocessing environment, the order of
	receive channels in the output file will be arbitrary.
	'''
	# Open the input WaveformSet, then create a corresponding output set
	wset = WaveformSet.fromfile(infile)
	# Attempt to truncate the input signals, if possible
	if nsamp is not None: wset.nsamp = nsamp
	# Create an empty waveform set to capture filtered output
	oset = WaveformSet.empty_like(wset)
	# The output always uses a float32 datatype
	oset.dtype = np.float32
	# Pull the bandwidth specifiers for the filter
	bstart, bend = filt[:2]
	# Try to pull the tailwidth, or use no roll-off
	try: tails = np.hanning(2 * filt[2])
	except IndexError: tails = None

	# Grab a list of transmit indices from the input
	txidx = wset.txidx

	for rxc in rxchans:
		# Pull the waveform header to copy to the output (ignore waveforms)
		hdr = wset.getrecord(rxc)[0]
		# Create an empty record in the output set to hold filtered waves
		oset.setrecord(hdr)

		for txc in txidx:
			# Pull the waveform the the Tx-Rx pair
			wave = wset.getwaveform(rxc, txc)
			# Set output to filtered waveform (force type conversion)
			owave = wave.bandpass(bstart, bend, tails, dtype=oset.dtype)
			oset.setwaveform(rxc, txc, owave)

	# Acquire the lock (if possible) and write the new records to the output
	try: lock.acquire()
	except AttributeError: pass

	oset.store(outfile, append=True)

	try: lock.release()
	except AttributeError: pass


def mpwavefilt(infile, filt, nproc, outfile, nsamp=None):
	'''
	Subdivide, along receive channels, the work of wavefilt() among
	nproc processes to bandpass filter the habis.formats.WaveformSet stored
	in infile into a WaveformSet file that will be written to outfile.

	The output file will be overwritten if it already exists.

	If nsamp is not None, it specifies a number of samples to which all
	input and output waveforms will be truncated. If nsamp is None, the
	length encoded in the input WaveformSet will be used.
	'''
	# Copy the input header to output and get receive-channel indices
	wset = WaveformSet.fromfile(infile)

	# Make sure the set can be truncated as desired
	if nsamp is not None:
		try:
			wset.nsamp = nsamp
		except ValueError:
			print >> sys.stderr, 'ERROR: could not truncate input waveforms'
			return

	copyhdr(outfile, wset, dtype=np.float32)
	rxidx = wset.rxidx

	# Delete the waveform set to close the memory-mapped input file
	del wset

	# Create a lock for the output file
	lock = multiprocessing.Lock()

	# Spawn the desired processes to perform the cross-correlation
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Assign a meaningful process name
			procname = process.procname(i)
			# Stride the recieve channels
			args = (infile, rxidx[i::nproc], filt, outfile, lock)
			pool.addtask(target=wavefilt, name=procname, args=args)

		pool.start()
		pool.wait()


def filterEngine(config):
	'''
	With multiple processes, perform bandpass filtering on a multiple
	habis.formats.WaveformSet objects stored in a collection of files.
	'''
	try:
		datafiles = config.getlist('filter', 'datafile')
		outfiles = config.getlist('filter', 'outfile')
		if len(datafiles) < 1 or len(datafiles) != len(outfiles):
			err = 'Keys datafile and outfile must have same length and at least one record'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify datafile and outfile in [filter]'
		raise HabisConfigError.fromException(err, e)

	try:
		nsamp = config.getint('filter', 'nsamp', failfunc=lambda: None)
	except Exception as e:
		err = 'Invalid specification of optional nsamp in [filter]'
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the number of processes to use (optional)
		nproc = config.getint('general', 'nproc',
				failfunc=process.preferred_process_count)
	except Exception as e:
		err = 'Invalid specification of optional nproc in [general]'
		raise HabisConfigError.fromException(err, e)

	try:
		filt = config.getlist('filter', 'filter', mapper=int)
		if len(filt) < 2 or len(filt) > 3:
			err = 'Specification of filter contains improper parameters'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify filter in [filter]'
		raise HabisConfigError.fromException(err, e)

	for datafile, outfile in zip(datafiles, outfiles):
		print 'Filtering data file', datafile
		mpwavefilt(datafile, filt, nproc, outfile, nsamp)


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
	filterEngine(config)
