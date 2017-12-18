#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, itertools, numpy as np

from argparse import ArgumentParser

from mpi4py import MPI

from pycwp import mio
from habis.habiconf import HabisConfigParser, HabisConfigError
from habis.formats import loadkeymat, savez_keymat
from habis.pathtracer import PathTracer, TraceError

if __name__ == '__main__':
	parser = ArgumentParser(description='Compute bent-ray as well as compensated '
			'and uncompensated straight-ray integrals through an image')

	parser.add_argument('-t', '--trmap', action='store_true',
			help='Treat target map as an r-[t] map instead of coordinates')
	parser.add_argument('-s', '--nostraight',
			action='store_true', help='Disable straight-ray tracing')
	parser.add_argument('-b', '--nobent',
			action='store_true', help='Disable bent-ray tracing')

	parser.add_argument('targets', type=str,
			help='Target map, either an r-[t] map or coordinate list')
	parser.add_argument('elements', type=str,
			help='A key-matrix of source (element) coordinates')
	parser.add_argument('tracer', type=str,
			help='HABIS configuration file containing pathtracer config')
	parser.add_argument('slowness', type=str,
			help='Numpy npy file containing the slowness image')
	parser.add_argument('output', type=str,
			help='Integral output, keymap or binary matrix (with -c)')

	args = parser.parse_args(sys.argv[1:])

	# Load the tracer configuration
	try: config = HabisConfigParser(args.tracer)
	except Exception as e:
		err = f'Unable to load configuration {args.tracer}'
		raise HabisConfigError.fromException(err, e)

	tracer = PathTracer.fromconf(config)

	# Load the element coordinates and target list
	elements = loadkeymat(args.elements)

	if args.trmap: targets = loadkeymat(args.targets)
	else: targets = mio.readbmat(args.targets)

	print(args)
