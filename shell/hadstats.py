#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, os, sys, getopt, cPickle

from numpy import fft, linalg as la

from itertools import izip

import multiprocessing, Queue

from pycwp import process, stats

from habis.habiconf import matchfiles, buildpaths
from habis.formats import WaveformSet
from habis.sigtools import Waveform, Window

def usage(progname=None, fatal=False):
	if progname is None: progname = sys.argv[0]
	binfile = os.path.basename(progname)
	print >> sys.stderr, 'USAGE: %s [-p p] [-q q] [-d d] [-n n] -o output -s <single-tx> <decoded WaveformSets>' % binfile
	print >> sys.stderr, '''
  Compare Hadamard-decoded and single-transmission WaveformSets.

  REQUIRED ARGUMENTS:
  -s: Search the provided directory for single-transmission WaveformSets
  -o: Provide a path for writing output using pickle.dump()

  OPTIONAL ARGUMENTS:
  -p: Use p processors (default: all available processors)
  -q: Set the width of the "quiet window" over which to estimate the SNR
  -n: Set the oversampling rate with which waveforms will be aligned
  -d: Set the maximum allowable delay shift before forcing a nominal shift
	'''
	if fatal: sys.exit(fatal)


def mphadtest(nproc, *args, **kwargs):
	'''
	Subdivide, along receive channels, the work of hadtest() among nproc
	processes to compare Hadamard-decoded and single-transmission
	WaveformSet files. The results of each hadtest() are accummulated and
	returned as a single dictionary.

	The positional and keyward arguments are passed to hadtest(). Any
	'stride', 'start' or 'queue' kwargs will be overridden by internally
	generated values.
	'''
	if nproc == 1:
		# For a single process, don't spawn
		return hadtest(*args, **kwargs)

	# Add the stride to the kwargs
	kwargs['stride'] = nproc

	# Create a multiprocessing queue to allow comparison results to be returned
	queue = multiprocessing.Queue()
	kwargs['queue'] = queue

	# Span the desired processes to perform FHFFT
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Give each process a meaningful name
			procname = process.procname(i)
			# Note the starting index for this processor
			kwargs['start'] = i
			pool.addtask(target=hadtest, name=procname, args=args, kwargs=kwargs)
		pool.start()

		# Wait for all processes to respond
		results = {}
		responses = 0
		while responses < nproc:
			try:
				locres = queue.get(timeout=0.1)
			except Queue.Empty:
				pass
			else:
				responses += 1
				results.update(locres)

		pool.wait()

		return results


def hadtest(decfile, stxfile, **kwargs):
	'''
	For a WaveformSet file decfile containing Hadamard-decoded waveforms,
	and corresponding single-transmission waveforms in the WaveformSet file
	stxfile, perform optional bandpass filtering on the waves in both sets,
	then return a dictionary providing statistics on the two sets.

	Each key in the return dictionary is a receive-channel index, as a
	string. The value for each receive-channel index is a dictionary whose
	values are Numpy record arrays in transmission order. The records in
	the array are:

	* delay: The list of delays d.delay(s, osamp=osamp) between the decoded
	  (d) and single-transmission (s) waves for each transmission.

	* esnr: The error SNR (in dB). For each transmission with decoded (d)
	  and single-transmission (s) waveforms, the error is E = d - s and the
	  error SNR is var(E) / min(stats.rolling_variance(E, qper)).

	* dsnr: The SNR (in dB) of the decoded waveforms for each transmission,
	  computed as d.snr(qper) when d is a habis.sigtools.Waveform instance.

	* ssnr: The SNR (in dB) of the single-transmission waveforms for each
	  transmission, computed as for dsnr.

	*** NOTE: Signals d and s, and error E, must have zero mean for the SNR
	    values defined above to make sense. This can be enforced by
	    bandpass filtering to exclude a DC component.

	The kwargs contain optional values or default overrides:

	* qper (default: 100): The width, in samples, of a sliding window used
	  to identify the quietest portion of a signal for SNR comparisons.

	* osamp (default: 1): Set the oversampling rate with which delays
	  between single-transmission and decoded waveforms will be computed.

	* maxd (default: 10): Set the maximum allowable magnitude of calculated
	  delays between single-transmission and decoded waveforms. If a delay
	  exceeds maxd, esnr will be calculated without aligning the waveforms.

	* start (default: 0) and stride (default: 1): For an input WaveformSet
	  wset, process receive channels in wset.rxidx[start::stride].

	* queue (default: None): If not None, this object's put() method
	  will be called with the return dictionary as the sole argument.
	'''
	# Grab the result return queue
	queue = kwargs.pop('queue', None)

	# Grab the oversampling factor
	osamp = kwargs.pop('osamp', 1)

	# Grab the maximum allowable delay
	maxd = kwargs.pop('maxd', 10)

	# Grab striding information
	start = kwargs.pop('start', 0)
	stride = kwargs.pop('stride', 1)

	# Grab the quiet-window width
	qper = int(kwargs.pop('qper', 100))

	if len(kwargs):
		raise TypeError("Unrecognized keyword argument '%s'" % kwargs.iterkeys().next())

	# Open the inputs and ensure that both are compatible
	decset = WaveformSet.fromfile(decfile)
	stxset = WaveformSet.fromfile(stxfile)

	results = {}

	if len(set(decset.rxidx).symmetric_difference(set(stxset.rxidx))):
		try: queue.put(results)
		except AttributeError: pass
		raise IndexError('Decoded and single-transmission sets contain different receive indices')
	if len(set(decset.txidx).symmetric_difference(set(stxset.txidx))):
		try: queue.put(results)
		except AttributeError: pass
		raise IndexError('Decoded and single-transmission sets contain different receive indices')
	if decset.txgrps != stxset.txgrps:
		try: queue.put(results)
		except AttributeError: pass
		raise ValueError('Decoded and single-transmission sets contain different group configurations')

	txlist = sorted(decset.txidx)

	# Build the record data type
	rectype = np.dtype([(name, '<f4') 
		for name in ['delay', 'esnr', 'dsnr', 'ssnr']])
	chanrecs = np.zeros((len(decset.rxidx[start::stride]), len(txlist)), dtype=rectype)

	for chanrow, rxc in izip(chanrecs, decset.rxidx[start::stride]):
		# Grab the records for the receive channel
		dechdr, decdat = decset.getrecord(rxc, txlist)
		stxhdr, stxdat = stxset.getrecord(rxc, txlist)

		# Shift the data windows to 0 f2c
		dwin = Window(dechdr.win.start + decset.f2c, dechdr.win.length)
		swin = Window(stxhdr.win.start + stxset.f2c, stxhdr.win.length)

		# Grab the overlapping portion of the window
		cwin = Window(max(dwin.start, swin.start), end=min(dwin.end, swin.end))

		for chanrec, decrow, stxrow in izip(chanrow, decdat, stxdat):
			# Convert each record row to a Waveform
			decwave = Waveform(dwin.end, decrow, dwin.start)
			stxwave = Waveform(swin.end, stxrow, swin.start)

			# Figure the delay between elements
			delay = decwave.delay(stxwave, osamp=osamp)
			chanrec['delay'] = delay

			chanrec['dsnr'] = decwave.snr(qper)
			chanrec['ssnr'] = stxwave.snr(qper)

			# Limit allowable shift for alignments
			if abs(delay) <= maxd:
				stxwave = stxwave.shift(delay)

			# Calculate the error SNR
			dwave = (decwave - stxwave).window(cwin)
			npwr = min(stats.rolling_variance(dwave._data, qper))
			chanrec['esnr'] = 10 * np.log10(np.var(dwave._data) / npwr)
			
		results[rxc] = chanrow

	try: queue.put(results)
	except AttributeError: pass

	return results


if __name__ == '__main__':
	# Set default options
	nprocs = process.preferred_process_count()
	outfile = None
	stxpath = None

	# Build optional or default overrides for hadtest
	kwargs = {}

	optlist, args = getopt.getopt(sys.argv[1:], 'hs:o:p:q:n:d:')

	for opt in optlist:
		if opt[0] == '-s':
			stxpath = opt[1]
		elif opt[0] == '-o':
			outfile = opt[1]
		elif opt[0] == '-p':
			nprocs = int(opt[1])
		elif opt[0] == '-q':
			kwargs['qper'] = int(opt[1])
		elif opt[0] == '-n':
			kwargs['osamp'] = int(opt[1])
		elif opt[0] == '-d':
			kwargs['maxd'] = int(opt[1])
		else:
			usage(fatal=True)

	if outfile is None or stxpath is None:
		print >> sys.stderr, 'ERROR: arguments -o and -s are required'
		usage(fatal=True)

	try: decfiles = matchfiles(args)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(fatal=True)

	try:
		stxfiles = buildpaths(decfiles, stxpath)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(fatal=True)

	# Accumulate results from all files
	results = {}

	for decfile, stxfile in zip(decfiles, stxfiles):
		print 'Comparing data files', decfile, stxfile
		fres = mphadtest(nprocs, decfile, stxfile, **kwargs)
		results.update(fres)

	cPickle.dump(results, open(outfile, 'wb'), protocol=2)
