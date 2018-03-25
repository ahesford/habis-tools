#!/usr/bin/env python

import os, sys, getopt
import numpy as np

from argparse import ArgumentParser

from random import sample

from multiprocessing import Pool

from habis.sigtools import WaveformMap
from habis.formats import loadkeymat
from habis.habiconf import matchfiles, numrange

def _checkoutdir(outspec):
	'''
	Check for existence of any directory portion in outspec.
	'''
	destdir = os.path.dirname(outspec)
	if destdir and not os.path.isdir(destdir):
		raise IOError('Destination %s is not a directory' % destdir)


def trextract(wmap, trmap=None, random=None):
	'''
	Extract waveforms for arbitrary transmit-receive pairs from a
	WaveformMap wmap, yielding each waveform as a tuple ((t, r), waveform)
	through a generator.

	If trmap is not None, it should be a map from receive indices to lists
	of desired transmit indices that are desired in the output.
	Transmit-receive pairs in trmap that do not exist in wmap are silently
	ignored. If trmap is None or empty, it is constructed to correspond
	bijectively to all transmit-receive pairs in wmap.

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
	# Convert the trmap to a set of pairs and count the total
	if trmap: trmap = { (t, r) for r, tl in trmap.items() for t in tl }
	else: trmap = set(wmap)
	ntrm = len(trmap)

	if random:
		if random >= 1: random = float(random) / ntrm
		else: random = float(random)
		if not 0 < random < 1:
			raise ValueError(f'Value "random" must be in range (0, {ntrm})')

	# Determine a local portion of the trmap
	desired = trmap.intersection(wmap)
	# Randomly sample the list as desired
	if random: desired = sample(desired, int(len(desired) * random))

	# Yield each value
	for key in desired: yield key, wmap[key]


if __name__ == '__main__':
	parser = ArgumentParser(description='Extract subsets of WaveformMap files')

	parser.add_argument('-m', '--trmap', type=str, default=None,
			help='Receive-to-transmit-list map of pairs to extact')
	parser.add_argument('-r', '--random', type=float,
			help='Randomly select pairs from map (fraction or approximate count)')
	parser.add_argument('-c', '--compression', type=str, default=None,
			help='Enable output compression (bzip2, lzma, deflate)')
	parser.add_argument('-b' ,'--backscatter', action='store_true',
			help='Extract backscatter waveforms, not arbitrary T-R pairs')

	parser.add_argument('-o', '--output', type=str, default=None,
			help='Output file (default: replace extension with extract.wmz)')

	parser.add_argument('inputs', type=str, nargs='+',
			help='Input WaveformMap files from which to extract')

	args = parser.parse_args(sys.argv[1:])

	# Try to read all input WaveformMap files
	infiles = matchfiles(args.inputs)

	# Read a defined receive-to-transmit-list map
	if args.trmap: args.trmap = loadkeymat(args.trmap, scalar=False)

	# At first, clobber the output
	append = False

	for infile in infiles:
		wmap = WaveformMap.load(infile)

		# Build the appropriate subset of the WaveformMap
		if not args.backscatter: wvs = trextract(wmap, args.trmap, args.random)
		else: wvs = ((k, v) for k, v in wmap.items() if k[0] == k[1])
		omap = WaveformMap(0, wvs)

		if args.output:
			# Save to common output and switch to append mode
			omap.store(args.output, compression=args.compression, append=append)
			append = True
		else:
			output = os.path.splitext(infile)[0] + 'extract.wmz'
			omap.store(output, compression=args.compression, append=False)
