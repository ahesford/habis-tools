#!/usr/bin/env python

import os, sys, numpy as np
from habis.formats import WaveformSet
from habis.habiconf import matchfiles

import gzip, bz2

def usage(progname=None, fatal=True):
	progname = progname or sys.argv[0]
	print('USAGE: %s <inputs>' % progname, file=sys.stderr)
	sys.exit(int(fatal))


if __name__ == '__main__':
	if len(sys.argv) < 2:
		usage(sys.argv[0], True)

	try: files = matchfiles(sys.argv[1:])
	except IOError as e:
		print('ERROR:', e, file=sys.stderr)
		usage(sys.argv[0], True)

	for f in files:
		oname, fext = os.path.splitext(f)

		if fext.lower() == '.gz':
			oname = os.path.splitext(oname)[0]
			ff = gzip.open(f, 'rb')
		elif fext.lower() == '.bz2':
			oname = os.path.splitext(oname)[0]
			ff = bz2.open(f, 'rb')
		else: ff = open(f, 'rb')

		oname += '.tgc.txt'

		wset = WaveformSet.fromfile(ff, header_only=True)

		try:
			tgc = wset.context['tgc']
		except (KeyError, AttributeError):
			print('File', f, 'specifies no TGC bytes')
		except Exception as e:
			print('ERROR:', e, file=sys.stderr)
		else:
			print('Extracting TGC parameters from file', f, 'to file', oname)
			np.savetxt(oname, tgc)
