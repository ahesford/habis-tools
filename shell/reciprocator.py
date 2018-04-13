#!/usr/bin/env python

# Copyright (c) 2018 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, os, sys, operator as op
from itertools import repeat
from random import sample as rsample
from collections import defaultdict

from argparse import ArgumentParser

import io
import weakref

from mpi4py import MPI

from habis.sigtools import WaveformMap, Window
from habis.habiconf import matchfiles
from habis.mpdfile import flocshare

_tx_rx_key_bits = 16


def _check_default_comm(comm):
	'''
	Return comm if comm is None, otherwise return MPI.COMM_WORLD
	'''
	return MPI.COMM_WORLD if comm is None else comm


def pack_key(t, r):
	'''
	Verify that both t and r are integers less than 2**16, then fold them
	into a single integer wherein the value t occupies the high-order 16
	bits and the value r occupies the low-order 16 bits.
	'''
	rxmask = (1 << _tx_rx_key_bits) - 1
	ti, ri = int(t), int(r)
	if not (t == ti and r == ri and 0 <= ti <= rxmask and 0 <= ri <= rxmask):
		raise ValueError(f'Keys must be integers in range [0, {rxmask}]')
	return (ti << _tx_rx_key_bits) | (ri & rxmask)


def unpack_key(key):
	'''
	Verify that key is an integer in a reasonable range, then unpack into a
	(t, r) pair as if key were produced with pack_key(t, r).
	'''
	maxkey = (1 << (_tx_rx_key_bits << 1)) - 1
	rxmask = (1 << _tx_rx_key_bits) - 1

	keyi = int(key)
	if not (key == keyi and 0 <= key <= maxkey):
		raise ValueError(f'Keys must be integers in range [0, {maxkey}]')

	r = key & rxmask
	t = key >> _tx_rx_key_bits

	return t, r


def findmaps(infiles, start=0, stride=1):
	'''
	Parse all of the WaveformMap instances encapsulated in infiles (a list
	of files or globs) to identify for each file a set of all keys (t, r)
	in that file.

	If (start, stride) is other than (0, 1), the keys in each file will be
	pared to only sorted(keys)[start::stride].

	A map from file names to the (optionally strided) set of pairs for each
	file is returned.
	'''
	pairmaps = { }
	for f in infiles:
		# Build the key generator
		keys = WaveformMap.generate(f, keys_only=True)
		if (start, stride) != (0, 1):
			# Sort for striding, if desired
			keys = sorted(keys)[start::stride]
		# Convert to set and store if nonempty
		keys = set(keys)
		if keys: pairmaps[f] = keys

	return pairmaps


def makewindower(window, *args, **kwargs):
	'''
	If window is not None, return a function that, when called with a
	Waveform as its only argument, invokes

		Waveform.window(win, *args, **kwargs),

	where win = { 'start': window[0], 'end': window[1] }.

	If window is None, just return None.
	'''
	if window is None: return None

	start, end = window
	win = { 'start': start, 'end': end }
	def windower(wave): return wave.window(win, *args, **kwargs)
	return windower


def loadlocalmaps(infiles, windower, *args, **kwargs):
	'''
	Invoke findmaps(*args, **kwargs) to identify a map from file names for
	WaveformMap serializations to desired keys for that file, then load
	each file and extract the subset of the contained WaveformMap
	corresponding to those keys.

	If window is not None, it should be a callable that will be applied to
	each Waveform before it is added to the map.
	'''
	wmap = WaveformMap()
	for f, pairs in findmaps(infiles, *args, **kwargs).items():
		# Define a filter to only load locally assigned keys
		for key, wave in WaveformMap.generate(f):
			if key not in pairs: continue
			if windower: wave = windower(wave)
			wmap[key] = wave

	return wmap


def homogenize(keymap, tolerance=1):
	'''
	Given a map keymap from ranks to sets of keys assigned to that rank,
	continually take a subset of the largest rank assignment and assign the
	subset to the smallest assignment until the difference in numbers of
	keys between the largest and smallest assignment is no larger than
	tolerance.

	If the same value is associated with more than one key, the value will
	be randomly assigned to one of the associated keys before
	homogenization is performed.
	'''
	if tolerance < 1:
		raise ValueError('Nonuniformity tolerance must be at least 1')

	# Copy the keymap as is
	keymap = { k: set(v) for k, v in keymap.items() }

	# Invert the key map in random order to scramble lists of duplicates
	invkmap = defaultdict(list)
	for k in rsample(list(keymap.keys()), len(keymap)):
		for vv in keymap[k]: invkmap[vv].append(k)

	# Remove duplicates
	for v, k in invkmap.items():
		# List of keys is already randomly ordered
		for key in k[1:]:
			try: keymap[key].remove(v)
			except KeyError: pass

	# Count the assignment for key k
	count = lambda k: len(keymap[k])
	while True:
		# Find the largest and smallest assignments
		largest = max(keymap, key=count, default=None)
		smallest = min(keymap, key=count, default=None)

		if largest is None or smallest is None: break

		lgct = count(largest)
		smct = count(smallest)

		# Terminate if the nonuniformity is close enough
		if lgct - smct <= tolerance: break

		# Reassign half the difference to equalize
		while count(largest) > count(smallest) + tolerance:
			keymap[smallest].add(keymap[largest].pop())

	return keymap


def equivkey(key, groupsize=1):
	'''
	For an integer key as produced by pack_key, unpack the key to a (t, r)
	pair, then compute the equivalence key

		(min(t,r) // groupsize, max(t,r) // groupsize),

	and return the equivalence after packing with pack_key.
	'''
	t, r = unpack_key(key)
	tt, rr = min(t,r) // groupsize, max(t,r) // groupsize
	return pack_key(tt, rr)


def keyroute(mykeys, groupsize=1, gcomm=None, lcomm=None):
	'''
	On the given global MPI communicator gcomm (or MPI.COMM_WORLD if None)
	and a local communicator lcomm that represents ranks on the same
	physical hardware (which will be created if None), accumulate a global
	map between ranks in gcomm and keys controlled by that rank. The input
	mykeys should provide an iterator or collection of all keys available
	to the rank.

	For the purposes of equivalent distribution, keys (t, r) will be
	considered equivalent (and so only assigned to a single rank) if they
	have the same value for equivkey(pack_key(t,r), groupsize).

	Keys are assigned approximately uniformly across each of the unique
	communicators in lcomm. An attempt is made to assign locally-available
	keys to the local rank, but this cannot be guaranteed.
	'''
	# Make sure we have local and global communicators
	gcomm = _check_default_comm(gcomm)
	if lcomm is None: lcomm = gcomm.Split_type(MPI.COMM_TYPE_SHARED)

	grank, gsize = gcomm.rank, gcomm.size
	lrank, lsize = lcomm.rank, lcomm.size

	# Create a communicator of local roots (other get a null communicator)
	rcomm = gcomm.Split(0 if not lrank else MPI.UNDEFINED)

	# Pack the locally available keys
	mykeys = set(pack_key(*key) for key in mykeys)

	# Collect on the local root the locally available keys
	lakeys = lcomm.gather(mykeys)

	if not lrank:
		# Coalesce locally available equivalence keys
		lakeys = { k for lk in lakeys for k in lk }

		# Map equivalence keys to locally available keys
		eq2key = defaultdict(set)
		for key in lakeys:
			eqk = equivkey(key, groupsize)
			eq2key[eqk].add(key)

		# No longer need locally available keys
		lakeys = None

		# Equalize equivalence keys on all local communicators
		eqkeymap = rcomm.gather(set(eq2key.keys()))
		# Homogenize the distribution of equivalence keys
		if not rcomm.rank: eqkeymap = homogenize(dict(enumerate(eqkeymap)))
		eqkeymap = rcomm.bcast(eqkeymap)

		# Compile maps of outgoing waveforms
		keyexch = { }
		for i, eqkm in eqkeymap.items():
			ckeys = eqkm.intersection(eq2key)
			keyexch[i] = { k for kk in ckeys for k in eq2key[kk] }

		# Exhange outgoing waveforms for incoming waveforms
		keyexch = rcomm.alltoall([keyexch[i] for i in range(rcomm.size)])
		# Flatten the incoming waveforms into reciprocal equivalences
		keyexch = sorted({equivkey(k) for kk in keyexch for k in kk })
		# Prepare to distribute the inbound keys to local ranks
		keyexch = [keyexch[i::lsize] for i in range(lsize)]
	else: keyexch = None

	# Distribute the inbound reciprocal keys among local ranks
	keyexch = lcomm.scatter(keyexch)

	# Gather assignments everywhere, unpack and unfold equivalence
	allkeys = { i: { (t, r) for k in v
				for tt, rr in [unpack_key(k)]
				for t, r in [(tt,rr), (rr,tt)] }
			for i, v in enumerate(gcomm.allgather(keyexch)) }

	return allkeys


def makesendbufs(wmap, destmap, nmax=10000):
	'''
	Given a WaveformMap wmap and a destmap, as produced by keyroute, that
	maps ranks in an MPI communicator to sets of keys in wmap that should
	be sent to that rank, prepare and return a map from destination ranks
	to a list of BytesIO buffers that each hold a serialized representation
	of subset (of at most nmax Waveforms) of wmap to be sent to that rank.
	'''
	# Assign the buffers to target ranks
	buffers = defaultdict(list)

	for rank, rkeys in destmap.items():
		remaining = list(rkeys.intersection(wmap))
		while remaining:
			# Build a submap to serialize
			rmap = WaveformMap((k, wmap[k]) for k in remaining[:nmax])
			# Serialize to a BytesIO stream
			bstr = io.BytesIO()
			rmap.store(bstr)
			# Append the buffer to the map
			buffers[rank].append(bstr)
			# Discard the serialized portion
			remaining = remaining[nmax:]

	return buffers


def swapbufsizes(buffers, comm=None):
	'''
	For a map buffers, prepared by makesendbufs, from ranks in the
	communicator comm (or MPI.COMM_WORLD if comm is None) to lists of
	BytesIO instances, use an all-to-all message to inform other ranks
	about the sizes of all WaveformMap byte streams to be sent from this
	rank. A map from ranks to lists of sizes of all incoming bytestreams
	for this rank will be returned.
	'''
	comm = _check_default_comm(comm)

	# Build a list of outbound buffer sizes
	sendsizes = [[b.tell() for b in buffers.get(i, [])] for i in range(comm.size)]
	# Send the outbound and get the inbound sizes
	recvsizes = comm.alltoall(sendsizes)
	return { rank: rs for rank, rs in enumerate(recvsizes) if rs }


def makerecvbufs(bufsizes):
	'''
	For a map between ranks in an MPI communicator to a list of message
	sizes in bytes, prepare a map from ranks to a list of BytesIO buffers
	that are sized to receive corresponding messages as listed in the
	array-valued entries of bufsizes (as prepared by swapbufsizes).

	BytesIO buffers are sized by opening an empty buffer, seeking to the
	final byte, and writing a null.
	'''
	buffers = { }
	for rank, sizes in bufsizes.items():
		bufs = [ ]
		for sz in sizes:
			bstr = io.BytesIO()
			if sz > 0:
				bstr.seek(sz - 1)
				bstr.write(b'\0')
			bufs.append(bstr)
		buffers[rank] = bufs

	return buffers


def postmessages(buffers, comm=None, send=True):
	'''
	Into the communicator comm (MPI.COMM_WORLD if None), post a send (if
	send is True) or receive (if send is False) of each buffer in buffers,
	a map from destination (when sending) or source (when receiving) rank
	to lists of BytesIO objects as prepared by makesendbufs or
	makerecvbufs.

	Messages are tagged by the index of the buffer in the list for each
	rank. 

	A list of MPI.Request instances corresponding to all posted messages is
	returned.
	'''
	comm = _check_default_comm(comm)

	if send:
		srcdest = 'dest'
		func = comm.Isend
	else:
		srcdest = 'source'
		func = comm.Irecv

	requests = [ ]
	for rank, bufs in buffers.items():
		rmtargs = { srcdest: rank }
		for i, buf in enumerate(bufs):
			req = func(buf.getbuffer(), tag=i, **rmtargs)
			requests.append(req)

	return requests


def procmessages(sendreqs, recvreqs, recvbufs):
	'''
	Enter a loop to process incoming messages and close out pending sends,
	yielding (t, r) pairs and Waveform records as they are received.

	The arguments sendreqs and recvreqs are, respectively, lists send and
	receive requests as prepared by postmessages. The argument recvbufs is
	a map from source ranks to lists of BytesIO buffers that will be
	populated with the incoming messages associated with recvreqs.

	No action is taken when send requests are ready, except to wait for
	their completion.
	'''
	# Track the number of receive requests to differentiate sends and receives
	nrecvs = len(recvreqs)

	# Lump all requests together for processing
	requests = recvreqs + sendreqs

	# Begin processing messages
	status = MPI.Status()
	while True:
		# Wait until a message can be processed
		idx = MPI.Request.Waitany(requests, status)
		if idx == MPI.UNDEFINED: break

		# Figure out the rank, tag and size of this message
		tag = status.tag

		if 0 <= idx < nrecvs:
			# Parse the incoming WaveformMap stream
			bstr = recvbufs[status.source][tag]
			bstr.seek(0)
			# Yield the keys and waveforms in turn
			yield from WaveformMap.generate(bstr)
			# Free buffer by closing the stream
			bstr.close()
		elif idx < 0 or idx >= len(requests):
			raise ValueError(f'Unexpected MPI request index {idx}')


def printroot(rank, *args, **kwargs):
	'''
	If rank is 0, invoke print(*args, **kwargs) and flush stdout.
	Otherwise, take no action.
	'''
	if rank: return
	print(*args, **kwargs)
	sys.stdout.flush()


def pairavg(left, right, osamp=0, clip=False):
	'''
	Compute and return the average of two waveforms left and right. If
	osamp is 0, the waveforms are simply averaged. Otherwise, if osamp is a
	positive integer, align the waveforms with the given oversampling rate
	prior to averaging.

	Note: alignment is done symmetrically: if D = left.delay(right, osamp)
	is the delay in right necessary to align with left, the average will be
	between waveforms left.shift(-D / 2) and right.shift(D / 2).

	If clip is True, the average waveform will be clipped to the narrowest
	data window that contains the data windows of the inputs. Otherwise,
	the data window may grow depending on any shifting performed to ensure
	alignment. (In particular, fractional-sample shifts will resample the
	signals across the entire signal window.)
	'''
	if not osamp:
		# Simple average
		avg = 0.5 * (left + right)
	else:
		# Shift waveforms towards center to align
		delay = left.delay(right, osamp) / 2
		avg = 0.5 * (left.shift(-delay) + right.shift(delay))

	if clip:
		# Find the window that contains both data windows
		start = min(left.datawin.start, right.datawin.start)
		end = max(left.datawin.end, right.datawin.end)
		win = Window(start, end=end, nonneg=True)
		avg = avg.window(win)

	return avg


if __name__ == '__main__':
	grank, gsize = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
	printroot(grank, 'Starting waveform reciprocator')

	parser = ArgumentParser(description='Exchange and average reciprocal waveforms')

	parser.add_argument('-s', '--osamp', type=int, default=0,
			help='Oversampling factor when aligning waveforms '
				'(when 0, waveforms are not aligned when averaging)')

	parser.add_argument('-o', '--output', default=None,
			help='Name of output file (required if more than one input)')

	parser.add_argument('-w', '--window', default=None,
			nargs=2, type=int, metavar=('START', 'END'),
			help='Zero samples outside range START to END')

	parser.add_argument('-T', '--tails', default=0, type=int,
			help='If windowing, apply rolloff of this width')

	parser.add_argument('-R', '--relative', choices=['datawin', 'signal'],
			default=None, help='Use relative window mode') 

	parser.add_argument('-c', '--clip', action='store_true',
			help='Tightly clip data window of average waveforms')

	parser.add_argument('-g', '--groupsize', default=64, type=int,
			help='Equalize output distribution '
				'by groups of GROUPSIZE elements')

	parser.add_argument('input', nargs='+', help='Inputs to process')

	args = parser.parse_args(sys.argv[1:])

	args.input = matchfiles(args.input, forcematch=False)
	printroot(grank, f'Identified {len(args.input)} input files')

	if args.output is None:
		if len(args.input) > 1:
			raise ValueError('Must specify output with more than one input')
		elif len(args.input) > 0:
			args.output = os.path.splitext(args.input[0])[0] + '.reciprocal.wmz'

	# Build a communicator for ranks on the same physical system
	lcomm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
	lrank, lsize = lcomm.rank, lcomm.size

	# Build the wave windower function
	windower = makewindower(args.window, args.tails, args.relative)

	# Find all locally available keys and load the local map
	printroot(grank, 'Loading local portions of waveform maps...')
	wmap = loadlocalmaps(args.input, windower, lrank, lsize)
	printroot(grank, f'Initial size of local map at rank {grank} is {len(wmap)}')

	# Find the destinations for local keys
	printroot(grank, 'Creating distribution maps...')
	destmap = keyroute(wmap, args.groupsize, MPI.COMM_WORLD, lcomm)

	# Remove the local rank from the destmap, but hold needed local keys
	mykeys = destmap.pop(grank)

	# Build the serialized representations of waveforms to send to other ranks
	printroot(grank, 'Serializing waveform maps for transmission...')
	sendbufs = makesendbufs(wmap, destmap)

	# Only locally needed keypairs need to remain
	for key in set(wmap).difference(mykeys): wmap.pop(key)
	printroot(grank, f'Pared size of local map at rank {grank} is {len(wmap)}')

	# Exchange the buffer sizes to find size of inbound messages
	printroot(grank, 'Exchanging message sizes...')
	rcvsizes = swapbufsizes(sendbufs)

	# Build the incoming receive buffers
	printroot(grank, 'Allocating receive buffers...')
	recvbufs = makerecvbufs(rcvsizes)

	# Post receives and sends
	printroot(grank, 'Posting receives and sends...')
	recvreqs = postmessages(recvbufs, send=False)
	sendreqs = postmessages(sendbufs, send=True)

	# Outbound buffers are captured by requests and no longer needed
	del sendbufs

	# Process the messages, adding waveforms to the local map
	printroot(grank, 'Collecting incoming waveforms...')
	wmap.update(procmessages(sendreqs, recvreqs, recvbufs))
	printroot(grank, f'Final size of local map at rank {grank} is {len(wmap)}')

	gnsize = MPI.COMM_WORLD.reduce(len(wmap))
	printroot(grank, f'{gnsize} waveforms scattered globally')

	# Build an output map
	omap = WaveformMap()
	while wmap:
		(t, r), left = wmap.popitem()
		try: right = wmap.pop((r, t))
		except KeyError: continue
		omap[min(t,r), max(t,r)] = pairavg(left, right, args.osamp, args.clip)

	gosize = MPI.COMM_WORLD.reduce(len(omap))
	printroot(grank, f'{gosize} reciprocal pairs averaged globally')

	# Write the output, serializing within local communicators
	for i in range(lsize):
		if i == lrank: omap.store(args.output, append=i)
		lcomm.Barrier()

	printroot(grank, 'End of control')
	MPI.COMM_WORLD.Barrier()
