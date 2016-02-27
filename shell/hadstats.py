#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, os, sys, getopt, cPickle

from numpy import fft, linalg as la

from itertools import izip

from scipy.signal import hilbert

import multiprocessing, Queue

from pycwp import process, stats

from habis.habiconf import matchfiles, buildpaths, HabisConfigParser, HabisConfigError, HabisNoOptionError
from habis.formats import WaveformSet, loadkeymat
from habis.sigtools import Waveform, Window

def usage(progname=None, fatal=False):
	if progname is None: progname = sys.argv[0]
	binfile = os.path.basename(progname)
	print >> sys.stderr, 'USAGE: %s <configuration>' % binfile
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
	and corresponding single-element waveforms in the WaveformSet file
	stxfile, perform optional bandpass filtering on the waves in both sets,
	then return a dictionary providing statistics on the two sets.

	Each key in the return dictionary is a receive-channel index, as a
	string. The value for each receive-channel index is a dictionary whose
	values are Numpy record arrays in transmission order. The records in
	the array are:

	* delay: The list of delays d.delay(s, osamp=osamp) between the decoded
	  (d) and single-element (s) waves for each transmission.

	* dppwr, sppwr: The peak power (on a linear scale) of the decoded (d)
	  and single-element (s) waveforms for each transmission, as the square
	  of the peak amplitude.

	* dnpwr, snpwr: The noise power (on a linear scale) of the
	  decoded (d) and single-element (s) waveforms for each transmission,
	  as the minimum variance over a sliding window of length qper.

	* epwr: The averaged power over a window around the expected peak
	  arrival time (or the entire signal if no expected arrival time is
	  provided), as var(E) for an error E = d - s.

	*** NOTE: Signals d, s and E, must have zero mean for the power as
	    defined above to make sense. This can be enforced by bandpass
	    filtering to exclude a DC component.

	The kwargs contain optional values or default overrides:

	* qper (default: 50): The width, in samples, of a sliding window used
	  to identify the quietest portion of a signal for SNR comparisons.

	* osamp (default: 1): The oversampling rate with which delays between
	  single-element and decoded waveforms will be computed.

	* maxdelay (default: 10): The maximum allowable delay between decoded
	  and single-element waveforms. If a delay exceeds maxdelay, epwr and
	  enpwr will be calculated without aligning the waveforms.

	* ref (default: None): If not None, should be the name of a file
	  specifying a reference waveform used to determine arrival times (and
	  alignment) for decoded and single-element waveforms. If None, the
	  delay will be determined by directly cross-correlating the waveforms.

	* peaks (default: None): If not None, should be a dictionary of kwargs
	  passed to Waveform.isolatepeak for every decoded and single-element
	  Waveform. An additional 'nearmap' key may be included to specify the
	  name of a file, loadable with habis.formats.loadkeymat, that
	  specifies a mapping between element indices and expected round-trip
	  delays (in samples, relative to 0 f2c) for monostatic reflections
	  from a target. The index argument to Waveform.isolatepeak will be
	  populated with 0.5 * (atmap[tx] + atmap[rx]) for a given Waveform
	  transmitted from element tx and received on element rx.

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
	maxdelay = kwargs.pop('maxdelay', 10)

	# Grab striding information
	start = kwargs.pop('start', 0)
	stride = kwargs.pop('stride', 1)

	# Grab the quiet-window width
	qper = int(kwargs.pop('qper', 50))

	# Load a cross-correlation reference if provided
	try: ref = Waveform.fromfile(kwargs.pop('ref'))
	except KeyError: ref = None

	# Grab the isolation parameters
	peaks = kwargs.pop('peaks', None)
	if peaks is not None:
		try: atmap = loadkeymat(peaks.pop('nearmap'))
		except KeyError: atmap = { }

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
		for name in ['delay', 'epwr', 'dppwr', 'dnpwr', 'sppwr', 'snpwr']])
	chanrecs = np.zeros((len(decset.rxidx[start::stride]), len(txlist)), dtype=rectype)

	for chanrow, rxc in izip(chanrecs, decset.rxidx[start::stride]):
		# Grab the records for the receive channel
		dechdr, decdat = decset.getrecord(rxc, txlist)
		stxhdr, stxdat = stxset.getrecord(rxc, txlist)

		# Calculate the peak powers
		chanrow['dppwr'][:] = np.max(np.abs(hilbert(decdat, axis=-1)), axis=-1)**2
		chanrow['sppwr'][:] = np.max(np.abs(hilbert(stxdat, axis=-1)), axis=-1)**2

		# Calculate the noise powers
		chanrow['dnpwr'][:] = np.min(stats.rolling_variance(decdat, qper), axis=-1)
		chanrow['snpwr'][:] = np.min(stats.rolling_variance(stxdat, qper), axis=-1)

		# Shift the data windows to 0 f2c
		dwin = dechdr.win.shift(decset.f2c)
		swin = stxhdr.win.shift(stxset.f2c)

		# Grab the overlapping portion of the window
		cwin = Window(max(dwin.start, swin.start), end=min(dwin.end, swin.end))

		for chanrec, decrow, stxrow, txc in izip(chanrow, decdat, stxdat, txlist):
			# Convert each record row to a Waveform
			decwave = Waveform(dwin.end, decrow, dwin.start)
			stxwave = Waveform(swin.end, stxrow, swin.start)
			
			if peaks is not None:
				# Find expected arrival time
				try: expk = 0.5 * (atmap[rxc] + atmap[txc])
				except KeyError: expk = None
				# Attempt to isolate peaks, if possible
				try: decwave = decwave.isolatepeak(expk, **peaks)
				except ValueError: pass
				try: stxwave = stxwave.isolatepeak(expk, **peaks)
				except ValueError: pass

			# Figure the delay between waveforms
			if ref is not None:
				decdelay = decwave.delay(ref, osamp=osamp)
				stxdelay = stxwave.delay(ref, osamp=osamp)
				delay = decdelay - stxdelay
			else:
				delay = decwave.delay(stxwave, osamp=osamp)

			chanrec['delay'] = delay

			# Limit allowable shift for alignments
			if abs(delay) <= maxdelay:
				stxwave = stxwave.shift(delay)

			# Calculate the error SNR
			dwave = (decwave - stxwave).window(cwin)
			chanrec['epwr'] = np.var(dwave._data)
			
		results[rxc] = chanrow

	try: queue.put(results)
	except AttributeError: pass

	return results


if __name__ == '__main__':
	# Build optional or default overrides for hadtest
	kwargs = {}

	if len(sys.argv) != 2:
		usage(sys.argv[0], fatal=True)

	try: config = HabisConfigParser(sys.argv[1])
	except:
		print >> sys.stderr, 'ERROR: could not load configuration file %s' % sys.argv[1]
		usage(sys.argv[0], fatal=True)

	# Configuration sections
	hsec = 'hadstats'

	try:
		decfiles = matchfiles(config.getlist(hsec, 'decoded'))
		if len(decfiles) < 1:
			err = "Key 'decoded' matches no input files"
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify at least one valid decoded file'
		raise HabisConfigError.fromException(err, e)

	try:
		stxfiles = buildpaths(decfiles, config.get(hsec, 'stxdir'))
	except Exception as e:
		err = 'Configuration must specify a valid location for single-element transmissions'
		raise HabisConfigError.fromException(err, e)

	try:
		outdir = config.get(hsec, 'outdir', default=None)
		outfiles = buildpaths(decfiles, outdir, 'hadstats.pickle')
	except Exception as e:
		err = 'Configuration must specify a valid location for output files'
		raise HabisConfigError.fromException(err, e)

	try:
		nproc = config.get('general', 'nproc', mapper=int,
				failfunc=process.preferred_process_count)
	except Exception as e:
		err = 'Invalid specification of optional nproc'
		raise HabisConfigError.fromException(err, e)

	try:
		kwargs['qper'] = config.get(hsec, 'noisewin', mapper=int)
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional noisewin'
		raise HabisConfigError.fromException(err, e)

	try:
		kwargs['osamp'] = config.get('sampling', 'osamp', mapper=int)
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional osamp'
		raise HabisConfigError.fromException(err, e)

	try:
		kwargs['maxdelay'] = config.get(hsec, 'maxdelay', mapper=int)
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional maxdelay'
		raise HabisConfigError.fromException(err, e)

	try:
		useref = config.get(hsec, 'useref', mapper=bool, default=True)
	except Exception as e:
		err = 'Invalid specification of optional useref'
		raise HabisConfigError.fromException(err, e)

	try:
		if not useref:
			raise HabisNoOptionError('Skip reference specification')
		kwargs['ref'] = config.get('measurement', 'reference')
	except HabisNoOptionError:
		print 'Delays will be estimated by direct cross-correlation'
	except Exception as e:
		err = 'Invalid specification of optional reference'
		raise HabisConfigError.fromException(err, e)
	else:
		print 'Delays will be estimated by cross-correlation with reference', kwargs['ref']

	try:
		kwargs['peaks'] = config.get(hsec, 'peaks')
	except HabisNoOptionError:
		pass
	except Exception as e:
		err = 'Invalid specification of optional peaks'
		raise HabisConfigError.fromException(err, e)


	for decfile, stxfile, outfile in zip(decfiles, stxfiles, outfiles):
		print 'Comparing data files (%s, %s) -> %s' % (decfile, stxfile, outfile)
		results = mphadtest(nproc, decfile, stxfile, **kwargs)
		cPickle.dump(results, open(outfile, 'wb'), protocol=2)
