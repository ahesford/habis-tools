#!/usr/bin/env python

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, os, argparse

from habis.formats import WaveformSet, loadkeymat


def nonzero(string):
	val = int(string)
	if not val: raise argparse.ArgumentTypeError('must be nonzero')
	return val

def subtridx(string):
	val = int(string)
	if not 0 <= val < 160:
		raise argparse.ArgumentTypeError('must be in range [0, 159]')
	return val


def txgrpshift(infile, outfile, subtri, shift, grpmap):
	'''
	Open the input file infile as a WaveformSet and apply the specified
	nonzero integer sample shift (positive is a time advance) to all
	transmissions from subtriangle subtri. The modified WaveformSet will be
	stored in outfile or, if outfile is None or an empty string, back to
	infile.
	
	If grpmap is not None, it specifies the location of a transmit-group
	file that is used to map subtriangle indices to transmission numbers.
	The grpmap must be provided if WaveformSet(infile).txgrps is not None
	and must be omitted if WaveformSet(infile).txgrps is None.
	'''
	# Validate numeric inputs
	try: subtri = subtridx(subtri)
	except (argparse.ArgumentTypeError, ValueError) as e:
		raise ValueError('Invalid subtriangle index: %s' % (e,))

	try: shift = nonzero(shift)
	except (argparse.ArgumentTypeError, ValueError) as e:
		raise ValueError('Invalid shift index: %s' % (e,))

	if not outfile:
		outfile = infile
		print('Will overwrite input file', infile)
	else:
		print('Will create new input file', outfile)

	wset = WaveformSet.load(infile)

	if wset.txgrps and not grpmap:
		raise ValueError('Argument "grpmap" required when input uses transmit groups')

	# Attempt to assign a group map
	if grpmap: wset.groupmap = loadkeymat(grpmap)

	# Map transmit subtriangle indices to record rows
	txrows = [ wset.tx2row(wset.element2tx(i))
			for i in range(64 * subtri, 64 * (subtri + 1)) ]

	# Perform the shifts for each receive-channel record
	for hdr, data in wset.allrecords():
		# Copy the tail of the data
		tail = data[txrows,shift:].copy()
		# Now do the cyclic shift
		data[txrows,-shift:] = data[txrows,:shift]
		data[txrows,:-shift] = tail
		# Store the record (no need to make another copy)
		wset.setrecord(hdr, data, copy=False)

	# Store the output
	wset.store(outfile)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
			description="Time shift transmissions on a single subtriangle")

	# Group map is required if input requires it
	parser.add_argument('-m', '--map', metavar='FILE', default=None, 
			help='Use txgrp map in %(metavar)s '
				'(required if input uses groups)')

	parser.add_argument('-s', '--subtri', type=subtridx, default=3,
			help='Index of transmit subtriangle to shift '
				'(default: %(default)d)')

	parser.add_argument('-x', '--ext', default=None, metavar='EXT',
			help='Replace input extension with %(metavar)s '
				'for output (ignored if output specified)')

	parser.add_argument('shift', type=nonzero,
			help='Number of samples to shift (positive is time advance)')

	parser.add_argument('input', help='Name of input WaveformSet')

	parser.add_argument('output', nargs='?', default=None,
			help='Name of output WaveformSet (omit to overwrite input)')

	args = parser.parse_args()

	if not args.output and args.ext:
		# Create an output by replacing input extension
		args.output = os.path.splitext(args.input)[0] + '.' + args.ext

	# Perform the shift
	txgrpshift(args.input, args.output, args.subtri, args.shift, args.map)
