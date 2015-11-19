#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, numpy as np, os, getopt

import multiprocessing, Queue

from pycwp import process
from habis.habiconf import matchfiles, buildpaths
from habis.formats import WaveformSet


def usage(progname=None, fatal=False):
	if progname is None: progname = os.path.basename(sys.argv[0])
	print >> sys.stderr, 'USAGE: %s [-g grpmap] [-l locmap] [-p nprocs]' % progname
	sys.exit(int(fatal))


def getchanlist(infiles, queue=None):
	'''
	Given a list infiles of names for WaveformSet files, return a
	dictionary mapping receive-channel header indices to a tuple (pos,
	txgrp) of corresponding attributes from the same header.

	If multiple files specify the same channel index, a ValueError will be
	raised.

	If queue is provided, the return dictionary will also be provided as
	the sole argument to queue.put().
	'''
	chanlist = {}

	for infile in infiles:
		# Open the input WaveformSet
		wset = WaveformSet.fromfile(infile)

		# Read the headers and update the channel list
		for hdr, _ in wset._records.itervalues():
			if hdr.idx in chanlist:
				raise ValueError('Channel index collision')
			chanlist[hdr.idx] = tuple(hdr.pos), hdr.txgrp and tuple(hdr.txgrp)

	try: queue.put(chanlist)
	except AttributeError: pass

	return chanlist


def mpchanlists(infiles, nproc=1):
	'''
	Subdivide the infiles lists among nproc processors, and merge the
	dictionaries of results.
	'''
	# Don't spawn more processes than files
	nproc = max(nproc, len(infiles))

	# Don't fork for a single input file
	if nproc < 2:
		return getchanlist(infiles,)

	# Create a Queue to collect results
	queue = multiprocessing.Queue()

	# Accumulate results here
	chanlist = {}

	# Spawn the desired processes to collect header information
	with process.ProcessPool() as pool:
		for i in range(nproc):
			# Assign a meaningful process name
			procname = process.procname(i)
			# Stride the input files
			args = (infiles[i::nproc], queue)
			pool.addtask(target=getchanlist, name=procname, args=args)

		pool.start()

		# Wait for all processes to respond
		responses = 0
		while responses < nproc:
			try:
				results = queue.get(timeout=0.1)
				responses += 1
			except Queue.Empty:
				pass
			else:
				for chan, rec in results.iteritems():
					if chan in chanlist:
						raise ValueError('Channel index collision')
					chanlist[chan] = rec

		# Allow all processe to finish
		pool.wait()

		return chanlist


if __name__ == '__main__':
	grpmap, locmap = None, None
	nprocs = process.preferred_process_count()

	optlist, args = getopt.getopt(sys.argv[1:], 'g:l:p:')

	for opt in optlist:
		if opt[0] == '-p':
			nprocs = int(opt[1])
		elif opt[0] == '-g':
			grpmap = opt[1]
		elif opt[0] == '-l':
			locmap = opt[1]
		else:
			usage(fatal=True)

	# Prepare the input and output lists
	try: infiles = matchfiles(args)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(fatal=True)

	# Get the channel list
	chanlist = mpchanlists(infiles, nprocs)

	# If no output files are specified, just print
	if not grpmap and not locmap:
		for idx, (pos, txgrp) in sorted(chanlist.iteritems()):
			print idx, pos, txgrp

	if grpmap:
		try:
			grps = [[i] + list(v[1]) for i, v in chanlist.iteritems()]
		except TypeError:
			print >> sys.stderr, 'Will not print group map when at least one channel has no group index'
		else:
			np.savetxt(grpmap, grps, fmt='%6d %6d %6d')

	if locmap:
		locs = [[i] + list(v[0]) for i, v in chanlist.iteritems()]
		np.savetxt(locmap, locs, fmt='%6d %14.8f %14.8f %14.8f')
