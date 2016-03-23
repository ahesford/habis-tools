#!/opt/python-2.7.9/bin/python

import os, sys, getopt
import numpy as np

from multiprocessing import Pool
from itertools import izip

from habis.formats import WaveformSet
from habis.habiconf import matchfiles, numrange

def usage(progname=None, fatal=True):
	progname = progname or sys.argv[0]
	print >> sys.stderr, 'USAGE: %s [-o outspec] [-d] [-m] [-r rx] [-t tx] <inputs>' % progname
	sys.exit(int(fatal))


def bsextract(infiles, outspec):
	'''
	Extract the backscatter waveforms from a sequence infiles of files,
	storing each in its own single-tx, single-rx WaveformSet.
	
	If outspec is provided, it should be a string that will be converted
	into a file name by calling outspec.format(rx) for each receive channel
	rx. If outspec is None or empty, output names will be generated by
	appending output suffixes to the input file names.
	'''
	if outspec:
		# Check any destination name in the outspec for sanity
		destdir = os.path.dirname(outspec)
		if destdir and not os.path.isdir(destdir):
			raise IOError('Destination %s is not a directory' % destdir)

	# Load all waveform sets
	wsets = [WaveformSet.fromfile(f) for f in infiles]

	for wset, f in izip(wsets, infiles):
		obase = outspec or (os.path.splitext(f)[0] + '.Element{0:05d}.backscat')
		print 'Extracting backscatter waves from file', f, 'to output spec', obase

		for rx in wset.rxidx:
			try: wf = wset.getwaveform(rx, rx, maptids=True)
			except KeyError: continue

			hdr = wset.getheader(rx).copy(txgrp=None)
			bsw = WaveformSet.fromwaveform(wf, hdr=hdr, tid=rx, f2c=wset.f2c)

			bsw.store(obase.format(rx))


def txrxextract(infiles, rxlist, txlist, maptids, output):
	'''
	For each file in the sequence infiles, extract, for each
	receive-channel record in rxlist, the transmit indices in txlist,
	storing the records in the specified output. The output will be
	overwritten if it exists. If maptids is True, the transmit indices are
	element indices that will be mapped to using the maptids option to
	WaveformSet.getwaveform. Otherwise, the raw transmit indices will be
	extracted.

	If rxlist or txlist is None, the output will contain all receive
	channels and all transmit indices in the input files.

	The output WaveformSet will not retain transmit-group information.
	'''
	if rxlist is not None: rxlist = set(rxlist)

	# Retain waveform sets with relevant receive channels
	wsets = []
	f2c, mxend, zero = float('inf'), 0, bool(0)
	for f in infiles:
		wset = WaveformSet.fromfile(f)
		if not rxlist or rxlist.intersection(wset.rxidx):
			wsets.append(wset)
			# Update f2c, acquisition window end, and data type
			f2c = min(f2c, wset.f2c)
			mxend = max(mxend, wset.f2c + wset.nsamp)
			zero += wset.dtype.type(0)

	if not len(wsets): return

	# Find the overall data type and number of samples
	nsamp = mxend - f2c
	dtype = np.dtype(type(zero))

	# Use a default txlist if none is provided
	if txlist is None: txlist = list(wsets[0].txidx)

	# Create the properly typed and windowed output set
	oset = WaveformSet(len(txlist), 0, nsamp, f2c, dtype)

	for wset in wsets:
		# Pull the relevant receive channels from each file
		if not rxlist:
			rxidx = sorted(wset.rxidx)
		else:
			rxidx = sorted(rxlist.intersection(wset.rxidx))

		for rx in rxidx:
			hdr, rec = wset.getrecord(rx, txlist, maptids=maptids)
			# Adjust the data window for the global f2c
			win = hdr.win.shift(wset.f2c - f2c)
			# Set the new output record
			oset.setrecord(hdr.copy(win=win, txgrp=None), rec, copy=False)

	oset.store(output)


if __name__ == '__main__':
	outspec = None
	maptids, diag = False, False
	rxlist, txlist = None, None

	optlist, args = getopt.getopt(sys.argv[1:], 'ho:dmr:t:')

	for opt in optlist:
		if opt[0] == '-h':
			usage(fatal=False)
		elif opt[0] == '-o':
			outspec = opt[1]
		elif opt[0] == '-d':
			diag = True
		elif opt[0] == '-m':
			maptids = True
		elif opt[0] == '-r':
			rxlist = numrange(opt[1])
		elif opt[0] == '-t':
			txlist = numrange(opt[1])
		else:
			usage(fatal=True)

	if not len(args):
		usage(fatal=True)

	try: infiles = matchfiles(args)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(sys.argv[0], True)

	if diag:
		bsextract(infiles, outspec)
	else:
		if not outspec:
			print >> sys.stderr, 'ERROR: output specification required'
			usage(sys.argv[0], True)

		txrxextract(infiles, rxlist, txlist, maptids, outspec)
