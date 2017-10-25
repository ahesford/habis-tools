#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, numpy as np, os, getopt
import multiprocessing


from collections import defaultdict

from habis.habiconf import matchfiles, buildpaths
from habis.formats import WaveformSet


def usage(progname, fatal=False):
	print('USAGE: %s [-c] [-z] [-o outpath] <inputs>' % progname, file=sys.stderr)
	sys.exit(int(fatal))


def concatenate(infiles, outfile, corrtx=False, defzero=False):
	'''
	Given a list infiles of file names for input habis.formats.WaveformSet
	objects, invoke WaveformSet.concatenate on the inputs. The argument
	defzero is passed as a keyword argument to concatenate(). The resulting
	WaveformSet is written to the file specified by outfile.

	If corrtx is True, the input names will be sorted lexicographically and
	the txstart for each input file will be replaced by a contiguous list
	of integers such that

		infiles[i].txidx = range(ltx, ltx + infiles[i].ntx),

	where ltx = 0 if i == 0 and ltx = infiles[i-1].txidx[-1] otherwise.

	No options are passed to WaveformSet.store when storing the output, so
	the merge will choose a default output format version and may be
	destructive. At a minimum, the context property of each input will not
	appear in the output.
	'''
	# Open the input WaveformSets (sort in case corrtx is desired)
	wsets = [WaveformSet.fromfile(infile) for infile in sorted(infiles)]

	if corrtx:
		# Correct the transmit indices
		ltx = 0
		for wset in wsets:
			wset.txstart = ltx
			ltx += wset.ntx

	WaveformSet.concatenate(*wsets, defzero=defzero).store(outfile)


if __name__ == '__main__':
	outpath = None
	kwargs = {}

	optlist, args = getopt.getopt(sys.argv[1:], 'ho:cz')

	for opt in optlist:
		if opt[0] == '-o':
			outpath = opt[1]
		elif opt[0] == '-c':
			kwargs['corrtx'] = True
		elif opt[0] == '-z':
			kwargs['defzero'] = True
		else:
			usage(sys.argv[0], fatal=True)

	# Prepare the input files and group by basename
	try: infiles = matchfiles(args)
	except IOError as e:
		print('ERROR:', e, file=sys.stderr)
		usage(sys.argv[0], fatal=True)

	ingroups = defaultdict(list)
	for infile in infiles:
		ingroups[os.path.basename(infile)].append(infile)

	try:
		# Map each basename to an ouptut file
		outnames = { k: buildpaths([k], outpath, 'cat.wset')[0] for k in ingroups }
	except IOError as e:
		print('ERROR:', e, file=sys.stderr)
		usage(sys.argv[0], fatal=True)

	for k, infiles in ingroups.items():
		if len(infiles) < 2:
			print('Ignoring file with basename', k)
			continue
		print('Concatenating files', k, 'to output', outnames[k])
		concatenate(infiles, outnames[k], **kwargs)
