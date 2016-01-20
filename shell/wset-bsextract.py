#!/opt/python-2.7.9/bin/python

import os, sys
from habis.formats import WaveformSet
from habis.habiconf import matchfiles, buildpaths

def usage(progname=None, fatal=True):
	progname = progname or sys.argv[0]
	print >> sys.stderr, 'USAGE: %s <inputs>' % progname
	sys.exit(int(fatal))


if __name__ == '__main__':
	if len(sys.argv) < 2:
		usage(sys.argv[0], True)

	try: files = matchfiles(sys.argv[1:])
	except IOError as e:
		print >> sys.stderr, 'ERROR:', e
		usage(sys.argv[0], True)

	for f in files:
		wset = WaveformSet.fromfile(f)
		obase = os.path.splitext(f)[0]
		print 'Extracting backscatter waves from file', f
		for rx in wset.rxidx:
			wset[rx,rx].store(obase + '.Element%05d.backscatter' % rx)
