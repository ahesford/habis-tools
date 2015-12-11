#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, numpy as np, os, getopt
import multiprocessing

from pycwp import process
from habis.habiconf import matchfiles, buildpaths
from habis.formats import WaveformSet
from habis.sigtools import Waveform


def usage(progname, fatal=False):
	print >> sys.stderr, 'USAGE: %s [-r rxlist] [-t txlist] [-p nprocs] [-n nsamp] [-o outpath] -f <start:end[:tails]> <inputs>' % progname
	sys.exit(int(fatal))


def wavefilt(infile, filt, outfile, rxchans=None, txchans=None,
		nsamp=None, lock=None, event=None):
	'''
	For the habis.formats.WaveformSet object stored in infile, successively
	filter all waveforms received by the channels specified in the sequence
	rxchans and transmitted by the channels specified in the sequence
	txchans using a bandpass filter (habis.sigtools.Waveform.bandpass). If
	rxchans is None, all receive channels are used. If txchans is None, it
	defaults to rxchans. Append the filtered waveforms to the specified
	file named by outfile, which will be created or truncated. All output
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
	# Create an empty waveform set to capture filtered output
	oset = WaveformSet.empty_like(wset)
	# The output always uses a float32 datatype and no transmission group
	oset.dtype = np.float32
	oset.txgrps = None

	if rxchans is None: rxchans = wset.rxidx
	if txchans is None: txchans = sorted(rxchans)

	# Map the transmit channels to transmission numbers
	try:
		gcount, gsize = wset.txgrps
	except TypeError:
		# With no grouping, the map is the identity
		txmap = dict((i, i) for i in txchans)
	else:
		txmap = { }
		for txch in txchans:
			try:
				hdr = wset.getrecord(txch)[0]
			except KeyError:
				raise KeyError('Could not determine transmission mapping for channel %d' % txch)
			try:
				i, g = hdr.txgrp
			except TypeError:
				raise ValueError('WaveformSet specifies Tx-group parameters, but record specifies none')
			txmap[txch] = i + g * gsize

	# Copy the transmit-channel map
	oset.txidx = txchans

	# Create the input file header, if necessary
	getattr(lock, 'acquire', lambda : None)()

	if not getattr(event, 'is_set', lambda : False)():
		oset.store(outfile, ver=(1,0))
		getattr(event, 'set', lambda : None)()

	getattr(lock, 'release', lambda : None)()

	for rxc in rxchans:
		# Pull the waveform header to copy to the output (ignore waveforms)
		hdr = wset.getrecord(rxc)[0]

		# Create an empty record in the output set to hold filtered waves
		oset.setrecord(hdr.copy(txgrp=None))

		for txc in txchans:
			# Pull the waveform for the Tx-Rx pair
			wave = wset.getwaveform(rxc, txmap[txc])
			# Set output to filtered waveform (force type conversion)
			owave = wave.bandpass(*filt, dtype=oset.dtype)
			oset.setwaveform(rxc, txc, owave)

	# Ensure that the header has been written
	getattr(event, 'wait', lambda : None)()

	# Write new records to output
	getattr(lock, 'acquire', lambda : None)()
	oset.store(outfile, append=True, ver=(1,0))
	getattr(lock, 'release', lambda : None)()


def mpwavefilt(infile, filt, nproc, outfile, rxchans=None, txchans=None, nsamp=None):
	'''
	Subdivide, along receive channels, the work of wavefilt() among
	nproc processes to bandpass filter the habis.formats.WaveformSet stored
	in infile into a WaveformSet file that will be written to outfile.

	If rxchans is None, it defaults to all receive channels in the file. If
	txchans is None, it defaults the rxchans.

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

	if rxchans is None: rxchans = wset.rxidx
	if txchans is None: txchans = sorted(rxchans)

	# Delete the waveform set to close the memory-mapped input file
	del wset

	# Create a lock and event for output synchronization
	lock = multiprocessing.Lock()
	event = multiprocessing.Event()

	# Spawn the desired processes to perform the cross-correlation
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Assign a meaningful process name
			procname = process.procname(i)
			# Stride the recieve channels
			rxidx = rxchans[i::nproc]
			args = (infile, filt, outfile, rxidx, txchans, nsamp, lock, event)
			pool.addtask(target=wavefilt, name=procname, args=args)

		pool.start()
		pool.wait()


if __name__ == '__main__':
	outpath, filt, nsamp = None, None, None
	rxchans, txchans = None, None
	nprocs = process.preferred_process_count()

	optlist, args = getopt.getopt(sys.argv[1:], 'ho:f:p:n:r:t:')

	for opt in optlist:
		if opt[0] == '-p':
			nprocs = int(opt[1])
		elif opt[0] == '-f':
			filt = tuple(int(s, base=10) for s in opt[1].split(':'))
		elif opt[0] == '-n':
			nsamp = int(opt[1])
		elif opt[0] == '-o':
			outpath = opt[1]
		elif opt[0] == '-r':
			rxchans = [int(s) for s in opt[1].split(',')]
		elif opt[0] == '-t':
			txchans = [int(s) for s in opt[1].split(',')]
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
		mpwavefilt(datafile, filt, nprocs, outfile, rxchans, txchans, nsamp)
