#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, os, sys, getopt, glob

from habis.formats import WaveformSet
from habis.sigtools import Window

from pycwp import mio, cutil

def usage(progname=None, fatal=False):
	if progname is None: progname = sys.argv[0]
	binfile = os.path.basename(progname)
	print >> sys.stderr, 'USAGE: %s [-o output] <wavesets>' % binfile
	if fatal: sys.exit(fatal)


def matstrip(infile, outfile,):
	'''
	Given an input WaveformSet infile, store all waveform data windows, in
	receive-channel order, in a 3-D binary matrix file outfile of dimension
	(ns, ntx, nrx).
	
	The number of samples ns is the minimum number required to store every
	data window in the file.
	'''
	wset = WaveformSet.fromfile(infile)

	# Determine a common data window for all records
	start, end = wset.nsamp, 0
	for hdr, _ in wset.allrecords():
		start = min(start, hdr.win.start)
		end = max(end, hdr.win.end)

	cwin = Window(start, end=end)
	buf = np.empty((wset.ntx, cwin.length), order='C', dtype=wset.dtype)

	# Allocate the output matrix
	matdims = (cwin.length, wset.ntx, wset.nrx)
	outslicer = mio.Slicer(outfile, dim=matdims, dtype=wset.dtype, trunc=True)

	for i, (hdr, data) in enumerate(wset.allrecords()):
		buf[:,:] = 0.

		try:
			# Find the overlapping data region
			bstart, dstart, length = cutil.overlap(cwin, hdr.win)
		except TypeError: pass

		# Copy data to the expanded buffer
		buf[:,bstart:bstart+length] = data[:,dstart:dstart+length]
		# Output slicer stores in FORTRAN order
		outslicer[i] = buf


if __name__ == '__main__':
	# Set default options
	outfiles = None

	optlist, args = getopt.getopt(sys.argv[1:], 'ho:')

	for opt in optlist:
		if opt[0] == '-o':
			outfiles = [opt[1]]
		else:
			usage(fatal=True)

	infiles = []
	for arg in args:
		if os.path.lexists(arg): infiles.append(arg)
		else: infiles.extend(glob.glob(arg))

	if outfiles is None:
		outfiles = [os.path.splitext(f)[0] + '.wset.mat' for f in infiles]

	if len(infiles) < 1:
		print >> sys.stderr, 'ERROR: No input files'
		usage(fatal=True)
	elif len(infiles) != len(outfiles):
		print >> sys.stderr, 'ERROR: output name count disagrees with input name count'
		usage(fatal=True)

	for infile, outfile in zip(infiles, outfiles):
		print 'Processing data file', infile
		matstrip(infile, outfile)
