#!/opt/python-2.7.9/bin/python

import os, sys, numpy as np
from habis.formats import WaveformSet
from habis.habiconf import matchfiles

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
		oname = os.path.splitext(f)[0] + '.tgc.txt'

		try:
			tgc = wset.context['tgc']
		except (KeyError, AttributeError):
			print 'File', f, 'specifies no TGC bytes'
		except Exception as e:
			print >> sys.stderr, 'ERROR:', e
		else:
			print 'Extracting TGC parameters from file', f, 'to file', oname
			np.savetxt(oname, tgc)
