#!/usr/bin/env python

import os, sys, getopt
import numpy as np

from random import sample

from multiprocessing import Pool


from habis.formats import WaveformSet, loadkeymat
from habis.habiconf import matchfiles, numrange

def usage(progname=None, fatal=True):
	progname = progname or sys.argv[0]
	print('USAGE: %s [-o outspec] [-r <random>] [-m <trmap>] <inputs>' % progname, file=sys.stderr)
	sys.exit(int(fatal))


def _checkoutdir(outspec):
	'''
	Check for existence of any directory portion in outspec.
	'''
	destdir = os.path.dirname(outspec)
	if destdir and not os.path.isdir(destdir):
		raise IOError('Destination %s is not a directory' % destdir)


def bsextract(infiles, outspec=None):
	'''
	Extract the backscatter waveforms from a sequence infiles of files,
	storing each in its own single-tx, single-rx WaveformSet.
	
	If outspec is provided, it should be a string that will be converted
	into a file name by calling outspec.format(rx) for each receive channel
	rx. If outspec is None or empty, output names will be generated by
	appending output suffixes to the input file names.
	'''
	if outspec: _checkoutdir(outspec)

	# Load all waveform sets
	wsets = [WaveformSet.fromfile(f) for f in infiles]

	for wset, f in zip(wsets, infiles):
		obase = outspec or (os.path.splitext(f)[0] + '.Backscatter{0:05d}.wset')
		print('Extracting backscatter waves from file', f, 'to output spec', obase)

		for rx in wset.rxidx:
			try: wf = wset.getwaveform(rx, rx, maptids=True)
			except KeyError: continue

			hdr = wset.getheader(rx).copy(txgrp=None)
			bsw = WaveformSet.fromwaveform(wf, hdr=hdr, tid=rx, f2c=wset.f2c)

			bsw.store(obase.format(rx))


def trextract(infiles, trmap, random=None, outspec=None):
	'''
	Extract waveforms for arbitrary transmit-receive pairs from a sequence
	infiles of files, storing each in its own single-tx, single-rx
	WaveformSet.

	The desired transmit-receive pairs should be provided in the map
	trmap, which maps receive indices to lists of desired transmit indices.
	Each pair in trmap will be extracted only if that pair exists in one of
	the input files; missing pairs will be silently ignored. Note that a
	(t, r) pair will be considered "missing" if the transmitter index
	cannot be mapped to a transmission index in an input file.
	
	If random is True, it should be a float in the range (0, 1) that
	specifies a random sampling fraction. The total number of (t, r) pairs
	that will actually be extracted (after discarding pairs that do not
	exist in the inputs) will be multiplied by random and converted to an
	int, and the resulting number of (t, r) pairs will be extracted from
	the inputs.

	If outspec is provided, it should be a string that will be converted
	into a file name by calling outspec.format(tx, rx) for each receive
	channel rx. If outspec is None or empty, output names will be generated
	by appending output suffixes to the input file names.
	'''
	if outspec: _checkoutdir(outspec)

	if random:
		random = float(random)
		if not 0 < random < 1:
			raise ValueError('Value of "random" must be in range (0, 1)')

	# Load all waveform sets
	wsets = [WaveformSet.fromfile(f) for f in infiles]

	for wset, f in zip(wsets, infiles):
		obase = outspec or (os.path.splitext(f)[0] + '.Tx{0:05d}.Rx{1:05d}.wset')
		print('Extracting Tx,Rx pairs from file', f, 'to output spec', obase)

		# Determine a local portion of the trlist
		rset = set(wset.rxidx)
		tset = set(wset.txidx)
		trlist = [ (t, r) for r in rset.intersection(trmap)
				for t in tset.intersection(trmap[r]) ]
		# Randomly sample the list as desired
		if random: trlist = sample(trlist, int(len(trlist) * random))
		# Sort the trlist by receiver first
		trlist.sort(key=lambda x: (x[1], x[0]))
		for t, r in trlist:
			try: wf = wset.getwaveform(r, t, maptids=True)
			except KeyError: continue

			hdr = wset.getheader(r).copy(txgrp=None)
			trw = WaveformSet.fromwaveform(wf, hdr=hdr, tid=t, f2c=wset.f2c)
			trw.store(obase.format(t, r))


if __name__ == '__main__':
	outspec = None
	trmap = None
	random = None

	optlist, args = getopt.getopt(sys.argv[1:], 'ho:m:r:')

	for opt in optlist:
		if opt[0] == '-h':
			usage(fatal=False)
		elif opt[0] == '-o':
			outspec = opt[1]
		elif opt[0] == '-m':
			trmap = loadkeymat(opt[1])
		elif opt[0] == '-r':
			random = float(opt[1])
		else:
			usage(fatal=True)

	if not len(args):
		usage(fatal=True)

	try: infiles = matchfiles(args)
	except IOError as e:
		print('ERROR:', e, file=sys.stderr)
		usage(sys.argv[0], True)

	if not trmap: bsextract(infiles, outspec)
	else: trextract(infiles, trmap, random, outspec)
