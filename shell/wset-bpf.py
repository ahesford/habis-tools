#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, numpy as np, os, getopt
import multiprocessing

from pycwp import process
from habis.habiconf import matchfiles, buildpaths, numrange
from habis.formats import WaveformSet
from habis.sigtools import Waveform


def usage(progname, fatal=False):
	print >> sys.stderr, 'USAGE: %s [-r rxlist] [-t txlist] [-z] [-p nprocs] [-n nsamp] [-o outpath] -f <start:end[:tails]> <inputs>' % progname
	sys.exit(int(fatal))


def wavefilt(infile, filt, outfile, rxchans=None, txchans=None,
		nsamp=None, debias=True, start=0, stride=1, lock=None, event=None):
	'''
	For the habis.formats.WaveformSet object in infile, successively filter
	all waves received by channels in the sequence rxchans[start::stride]
	and transmitted by the channels specified in the sequence txchans using
	a bandpass filter (habis.sigtools.Waveform.bandpass). If rxchans is
	None, all receive channels are used. If txchans is None, it defaults to
	rxchans. If txchans is "all" (regardless of case), every transmission
	index is used. Append the filtered waveforms to the specified file
	named by outfile, which will be created or truncated. All output
	waveforms will be of type float32.

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

	If lock is provided, it will be acquired and released by calling
	lock.acquire() and lock.release(), respectively, immediately prior to
	and following the append. If event is provided, its set() and wait()
	methods are used as a barrier prior to processing to ensure the
	multiprocess header is written.

	** NOTE **
	If this function is used in a multiprocessing environment, the order of
	receive channels in the output file will be arbitrary.
	'''
	# Open the input WaveformSet, then create a corresponding output set
	wset = WaveformSet.fromfile(infile)
	# Attempt to truncate the input signals, if possible
	if nsamp is not None: wset.nsamp = nsamp
	if rxchans is None: rxchans = wset.rxidx
	if txchans is None:
		txchans = sorted(rxchans)
	elif txchans.lower() == 'all':
		txchans = sorted(list(wset.txidx))

	# Create an empty waveform set to capture filtered output
	oset = WaveformSet.empty_like(wset)
	# Override transmit indices and data type
	oset.dtype = np.float32
	oset.txidx = txchans

	# Create the input file header, if necessary
	getattr(lock, 'acquire', lambda : None)()

	if not getattr(event, 'is_set', lambda : False)():
		oset.store(outfile)
		getattr(event, 'set', lambda : None)()

	getattr(lock, 'release', lambda : None)()

	for rxc in rxchans[start::stride]:
		# Read the record
		hdr, data = wset.getrecord(rxc)

		# Create an empty record in the output set to hold filtered waves
		oset.setrecord(hdr)

		for txc in txchans:
			# Pull the waveform for the Tx-Rx pair
			wave = Waveform(wset.nsamp, data[txc], hdr.win.start)
			# Debias before filtering to mitigate Gibbs phenomenon
			if debias: wave.debias()
			# Set output to filtered waveform (force type conversion)
			owave = wave.bandpass(*filt, dtype=oset.dtype)
			oset.setwaveform(rxc, txc, owave)

	# Ensure that the header has been written
	getattr(event, 'wait', lambda : None)()

	# Write new records to output
	getattr(lock, 'acquire', lambda : None)()
	oset.store(outfile, append=True)
	getattr(lock, 'release', lambda : None)()


def mpwavefilt(nproc, *args, **kwargs):
	'''
	Subdivide, along receive channels, the work of wavefilt() among
	nproc processes.
	'''
	if 'lock' in kwargs:
		raise TypeError('Keyword argument "lock" is forbidden')
	if 'event' in kwargs:
		raise TypeError('Keyword argument "lock" is forbidden')
	if 'stride' in kwargs:
		raise TypeError('Keyward argument "stride" is forbidden')
	if 'start' in kwargs:
		raise TypeError('Keyward argument "start" is forbidden')

	# Create a lock and event for output synchronization
	kwargs['lock'] = multiprocessing.Lock()
	kwargs['event'] = multiprocessing.Event()

	# Set the "stride" argument
	kwargs['stride'] = nproc

	# Spawn the desired processes to perform the cross-correlation
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Assign a meaningful process name
			procname = process.procname(i)
			# Stride the recieve channels
			kwargs['start'] = i
			pool.addtask(target=wavefilt, name=procname, args=args, kwargs=kwargs)

		pool.start()
		pool.wait()


if __name__ == '__main__':
	outpath, filt = None, None
	nprocs = process.preferred_process_count()

	# Optional arguments
	kwargs = { }

	optlist, args = getopt.getopt(sys.argv[1:], 'ho:f:p:n:r:t:z')

	for opt in optlist:
		if opt[0] == '-p':
			nprocs = int(opt[1])
		elif opt[0] == '-f':
			filt = tuple(int(s, base=10) for s in opt[1].split(':'))
		elif opt[0] == '-n':
			kwargs['nsamp'] = int(opt[1])
		elif opt[0] == '-o':
			outpath = opt[1]
		elif opt[0] == '-r':
			kwargs['rxchans'] = numrange(opt[1])
		elif opt[0] == '-t':
			if opt[1].lower() == 'all':
				kwargs['txchans'] = 'all'
			else:
				kwargs['txchans'] = numrange(opt[1])
		elif opt[0] == '-z':
			kwargs['debias'] = False
		else:
			usage(sys.argv[0], fatal=True)

	if filt is None:
		print >> sys.stderr, 'ERROR: must specify filter configuration'
		usage(sys.argv[0], fatal=True)

	# Prepare the input and output lists
	try: infiles = matchfiles(args)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(sys.argv[0], fatal=True)

	try: outfiles = buildpaths(infiles, outpath, 'bpf.wset')
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(sys.argv[0], fatal=True)

	# Process the waveforms
	for datafile, outfile in zip(infiles, outfiles):
		print 'Filtering data file', datafile, '->', outfile
		mpwavefilt(nprocs, datafile, filt, outfile, **kwargs)
