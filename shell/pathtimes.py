#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, itertools, numpy as np

from itertools import product as iproduct
from argparse import ArgumentParser

from mpi4py import MPI

import progressbar

from collections import defaultdict

from pycwp import mio
from habis.habiconf import HabisConfigParser, HabisConfigError
from habis.formats import loadkeymat, savez_keymat
from habis.pathtracer import PathTracer, TraceError

timetype = np.dtype([('t', '<i8'), ('r', '<i8'),
			('b', '<f8'), ('c', '<f8'), ('s', '<f8')])
jointype = np.dtype([('t', '<i8'), ('r', '<i8'),
			('x', '<f8'), ('y', '<f8'), ('z', '<f8')])

def tracetimes(tracer, s, elements, targets, trlist,
		skip_straight=False, skip_bent=False, quiet=True, joints=None):
	'''
	For a PathTracer tracer, a 3-D Numpy array s representing a slowness, a
	keymap of source coordinates elements, and a list of targets, trace
	indicated paths to determine straight-ray (if skip_straight is False)
	and bent-ray (if skip_bent is False) arrival times.

	If trlist is True, targets is interpreted as a list of (t, r) indices
	into the keymap elements. Otherwise, targets should be a Numpy array
	(or compatible sequence) of shape (N, 3) such that targets[i] contains
	the world coordinates for the i-th target.

	The return value is a Numpy record array of "timetype" records that
	each indicate transmit and receive indices and correpsonding times for
	bent-ray as well as compensated and uncompensated straight-ray times.

	If skip_straight is True, the straight-ray times will both be 0. If
	skip_bent is True, the bent-ray time will be 0. Both skip_straight and
	skip_bent cannot be True.

	Failure to trace a path will cause its time to be -1.

	If quiet is not True, a progress bar will be printed to show tracing
	progress.

	If joints is None, a second return value will be mapping from a (t, r)
	pair to a list of (x, y, z) tuples, each as coordinates of the N joints
	that describe the (N+1)-segment path, where joints do not count the
	location of transmitter t or receiver r. Single-segment paths, which
	have no joints, will not appear in the mapping.
	'''
	if skip_straight and skip_bent:
		raise ValueError('Cannot set both skip_straight and skip_bent to True')

	if joints and skip_bent:
		raise ValueError('Cannot set joints if skip_bent is True')

	if skip_straight: ntimes = 1
	elif skip_bent: ntimes = 2
	else: ntimes = 3

	if trlist:
		nrec = len(targets)
		itertr = iter(targets)
		targc = elements
	else:
		M, N = len(elements), len(targets)
		nrec = M * N
		itertr = iproduct(range(M), range(N))
		targc = targets

	times = np.zeros((nrec,), dtype=timetype)

	tracer.set_slowness(s)

	if not quiet: bar = progressbar.ProgressBar(max_value=nrec)
	else: bar = None

	if joints is not None: joints = { }

	for i, (t, r) in enumerate(itertr):
		src = elements[t]
		rcv = targc[r]

		record = [ t, r ]

		if not skip_bent:
			try: _, pts, tb = tracer.trace(src, rcv)
			except (ValueError, TraceError): tb = -1

			if joints is not None and len(pts) > 3:
				joints[t,r] = [ (x,y,z) for x,y,z in pts[1:-1] ]
		else: tb = 0
		record.append(tb)

		if not skip_straight:
			try: st = tracer.trace(src, rcv, intonly=True, mode='straight')
			except ValueError: st = (-1, -1)
		else: st = (0, 0)
		record.extend(st)

		times[i] = tuple(record)

		if not quiet: bar.update(i)

	if not quiet: bar.update(nrec)

	if joints is not None: return times, joints
	else: return times

if __name__ == '__main__':
	parser = ArgumentParser(description='Compute bent-ray as well as compensated '
			'and uncompensated straight-ray integrals through an image')

	parser.add_argument('-t', '--trlist', action='store_true',
			help='Treat target map as an r-[t] map instead of coordinates')
	parser.add_argument('-s', '--nostraight',
			action='store_true', help='Disable straight-ray tracing')
	parser.add_argument('-b', '--nobent',
			action='store_true', help='Disable bent-ray tracing')
	parser.add_argument('-q', '--quiet',
			action='store_true', help='Disable printing of status bar')
	parser.add_argument('-j', '--joints', type=str, default=None,
			help='Save interior joints in bent paths '
				'with at least 3 segments (requires bent tracing)')

	parser.add_argument('elements', type=str,
			help='A key-matrix of source (element) coordinates')
	parser.add_argument('targets', type=str,
			help='Target map, either an r-[t] map or coordinate list')
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

	if args.trlist:
		# Load the r-[t] keymap and flatten to a trlist
		targets = loadkeymat(args.targets)
		targets = [ (t, r) for r, tl in targets.items() for t in tl ]
	else: targets = mio.readbmat(args.targets)

	s = np.load(args.slowness)

	comm = MPI.COMM_WORLD
	rank, size = comm.rank, comm.size

	args.quiet = rank or args.quiet
	times = tracetimes(tracer, s, elements, targets[rank::size], args.trlist,
				args.nostraight, args.nobent, args.quiet, args.joints)
	if args.joints:
		# Expand two-value return when joints are desired
		times, joints = times
	else: joints = None

	# Gather the number of records at the root
	ntimes = len(times)
	counts = comm.gather(ntimes)

	# Flatten the joint mappings for collection with MPI
	if joints is not None:
		joints = np.array([(t, r, x, y, z) for (t, r), pts in joints.items()
					for x, y, z in pts], dtype=jointype)
		jcounts = comm.gather(len(joints))

	if not rank:
		# Store all accumulated rows
		rows = sum(counts)
		displs = [0]
		for ct in counts[:-1]: displs.append(displs[-1] + ct)
		alltimes = np.empty((rows,), dtype=timetype)

		if joints is not None:
			rows = sum(jcounts)
			jdispls = [0]
			for ct in jcounts[:-1]: jdispls.append(jdispls[-1] + ct)
			alljoints = np.empty((rows,), dtype=jointype)
	else:
		# Store only the local portion for accumulation
		rows = len(times)
		alltimes, displs = None, None
		if joints is not None: alljoints, jdispls = None, None

	# Build an MPI datatype for the record accumulation
	offsets = [timetype.fields[n][1] for n in timetype.names]
	mtypes = [MPI.LONG]*2 + [MPI.DOUBLE]*3
	mptype = MPI.Datatype.Create_struct([1]*len(mtypes), offsets, mtypes)
	mptype.Commit()

	if joints is not None:
		offsets = [jointype.fields[n][1] for n in jointype.names]
		mtypes = [MPI.LONG]*2 + [MPI.DOUBLE]*3
		jptype = MPI.Datatype.Create_struct([1]*len(mtypes), offsets, mtypes)
		jptype.Commit()

	# Accumulate all records on the root node
	comm.Gatherv([times, ntimes, mptype], [alltimes, counts, displs, mptype])
	# No need for MPI datatype anymore
	mptype.Free()

	if joints is not None:
		comm.Gatherv([joints, len(joints), jptype],
				[alljoints, jcounts, jdispls, jptype])
		jptype.Free()

	if not rank:
		if args.trlist:
			# Convert the record array to a keymap and save
			times = { (t, r): (b, c, s) for (t, r, b, c, s) in alltimes }
			savez_keymat(args.output, times)
		else:
			# Find indices to sort on transmit-receive pair
			indx = np.lexsort((alltimes['r'], alltimes['t']))
			# Rearrange the array according to the sort order
			np.take(alltimes, indx, out=alltimes)
			# Write the values, ignoring indices
			b = alltimes.view('float64').reshape(alltimes.shape + (-1,))[:,2:]
			mio.writebmat(b.astype('float32'), args.output)

		if args.joints and len(alljoints):
			# Convert flattened joint array to a list of points
			joints = defaultdict(list)
			for (t, r, x, y, z) in alljoints: joints[t,r].extend((x,y,z))
			savez_keymat(args.joints, joints)
