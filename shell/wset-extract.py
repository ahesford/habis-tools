#!/usr/bin/env python

import os, sys, getopt
import numpy as np

from argparse import ArgumentParser

from random import sample

from multiprocessing import Pool

from habis.sigtools import WaveformMap
from habis.formats import WaveformSet, loadkeymat
from habis.habiconf import matchfiles, numrange

def _checkoutdir(outspec):
	'''
	Check for existence of any directory portion in outspec.
	'''
	destdir = os.path.dirname(outspec)
	if destdir and not os.path.isdir(destdir):
		raise IOError('Destination %s is not a directory' % destdir)


def bsextract(wset):
	'''
	Extract the backscatter waveforms from a WaveformSet wset, returning
	the waves in a WaveformMap object.
	'''
	wmap = WaveformMap()
	for rx in wset.rxidx:
		try: wmap[rx,rx] = wset.getwaveform(rx, rx, maptids=True)
		except KeyError: continue
	return wmap


def trextract(wset, trmap, random=None):
	'''
	Extract waveforms for arbitrary transmit-receive pairs from a
	WaveformSet wset, returning the waves in a WaveformMap object.

	The desired transmit-receive pairs should be provided in the map
	trmap, which maps receive indices to lists of desired transmit indices.
	Each pair in trmap will be extracted only if that pair exists in one of
	the input files; missing pairs will be silently ignored. Note that a
	(t, r) pair will be considered "missing" if the transmitter index
	cannot be mapped to a transmission index in an input file.

	If random is True, it should be a float in the range (0, 1) or an
	integer no less than 1. If random is a float in (0, 1) range, it
	specifies a random sampling fraction. The total number of (t, r) pairs
	that will actually be extracted (after discarding pairs that do not
	exist in the inputs) will be multiplied by random and converted to an
	int, and the resulting number of (t, r) pairs will be extracted from
	the inputs. If random is at least 1, the random sampling fraction is
	computed by dividing the value of random by the total number of (t, r)
	pairs in trmap. In other words, if a collection of files contains all
	pairs in trmap, specifying an integer for random should result in a
	total number of extracted waveforms that approximately equals the value
	of random.
	'''
	# Find the maximum number of (t, r) pairs
	ntrm = sum(len(v) for v in trmap.values())
	if random:
		if random >= 1: random = float(random) / ntrm
		else: random = float(random)
		if not 0 < random < 1:
			raise ValueError(f'Value of "random" must be in range (0, {ntrm})')

	wmap = WaveformMap()

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
		try: wmap[t,r] = wset.getwaveform(r, t, maptids=True)
		except KeyError: continue

	return wmap


if __name__ == '__main__':
	parser = ArgumentParser(description='Extract waveforms from WaveformSet files')

	parser.add_argument('-m', '--trmap', type=str, default=None,
			help='Receive-to-transmit-list map of pairs to extact')
	parser.add_argument('-r', '--random', type=float,
			help='Randomly select pairs from map (fraction or approximate count)')
	parser.add_argument('-g', '--groupmap', type=str, default=None,
			help='Group map to assign to each WaveformSet to map extractions')
	parser.add_argument('-c', '--compression', type=str, default=None,
			help='Enable output compression (bzip2, lzma, deflate)')

	parser.add_argument('-o', '--output', type=str, default=None,
			help='Output file (default: replace extension with extract.wvzip)')

	parser.add_argument('inputs', type=str, nargs='+',
			help='Input WaveformSet files from which to extract')

	args = parser.parse_args(sys.argv[1:])

	# Try to read all input WaveformSets
	infiles = matchfiles(args.inputs)

	if args.groupmap:
		args.groupmap = loadkeymat(args.groupmap)
	if args.trmap:
		args.trmap = loadkeymat(args.trmap, scalar=False)

	# At first, clobber the output
	append = False

	for infile in infiles:
		wset = WaveformSet.load(infile)

		# Assign a group map as appropriate
		if args.groupmap is not None:
			wset.groupmap = groupmap

		# Build the WaveformMap
		if not args.trmap: wmap = bsextract(wset)
		else: wmap = trextract(wset, args.trmap, args.random)

		if args.output:
			# Save to common output and switch to append mode
			wmap.store(args.output, compression=args.compression, append=append)
			append = True
		else:
			output = os.path.splitext(infile)[0] + 'extract.wvzip'
			wmap.store(output, compression=args.compression, append=False)
