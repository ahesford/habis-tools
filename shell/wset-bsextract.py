#!/opt/python-2.7.9/bin/python

import os, sys, getopt
from habis.formats import WaveformSet
from habis.habiconf import matchfiles

def usage(progname=None, fatal=True):
	progname = progname or sys.argv[0]
	print >> sys.stderr, 'USAGE: %s [-p prefix] <inputs>' % progname
	sys.exit(int(fatal))


if __name__ == '__main__':
	prefix = None

	optlist, args = getopt.getopt(sys.argv[1:], 'hp:')

	for opt in optlist:
		if opt[0] == '-h':
			usage(fatal=False)
		elif opt[0] == '-p':
			prefix = opt[1]
		else:
			usage(fatal=True)

	if not len(args):
		usage(fatal=True)

	try: infiles = matchfiles(args)
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(sys.argv[0], True)

	if prefix:
		# Check any destination name in the prefix for sanity
		destdir = os.path.dirname(prefix)
		if destdir and not os.path.isdir(destdir):
			raise IOError('Destination %s is not a directory' % destdir)

	for f in infiles:
		wset = WaveformSet.fromfile(f)

		obase = prefix or os.path.splitext(f)[0]
		print 'Extracting backscatter waves from file', f, 'to files %s.*' % obase

		for rx in wset.rxidx:
			try: wf = wset.getwaveform(rx, rx, maptids=True)
			except KeyError: continue

			hdr = wset.getheader(rx).copy(txgrp=None)
			bsw = WaveformSet.fromwaveform(wf, hdr=hdr, tid=rx, f2c=wset.f2c)

			bsw.store(obase + '.Element%05d.backscatter' % rx)
