#!/usr/bin/env python

import os, sys, itertools, numpy as np
import multiprocessing, Queue

from itertools import izip

from pycwp import process

from habis.formats import WaveformSet
from habis.sigtools import Waveform
from habis.habiconf import HabisConfigError, HabisConfigParser

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def wavepaths(elements, reflectors, nargs={}):
	'''
	Given a list of elements grouped by facet, and a collection of
	reflectors, return a tuple (distances, angles) that indicates the
	distances from each element to each reflector and the angle between the
	propagation direction and the element's directivity axis (the normal to
	the facet). The normal is found using habis.facet.lsqnormal, and the
	optional nargs is a dictionary of kwargs to pass after the facet
	element coordinates argument.

	The argument elements should be a sequence of ndarrays such that
	elements[i] is an N[i]-by-3 array of (x, y, z) coordinates for each of
	N[i] elements in the i-th facet, and reflectors should be an Nr-by-3
	array of (x, y, z) center coordinates for each of Nr reflectors.

	The outputs will be lists of ndarrays such that distances[i] and
	angles[i] are N[i]-by-Nr maps of propagation distances or angles,
	respectively, such that entry (j, k) is the measure from element j to
	reflector k. These lists are suitable for passing to np.concatenate(),
	with axis=0, to produce composite element-to-reflector maps.
	'''
	from numpy.linalg import norm
	from habis.facet import lsqnormal
	# Compute the propagation vectors and normalize
	directions = [reflectors[np.newaxis,:,:] - el[:,np.newaxis,:] for el in elements]
	distances = [norm(dirs, axis=-1) for dirs in directions]
	directions = [dirs / dists[:,:,np.newaxis] 
			for dirs, dists in izip(directions, distances)]

	# The normal should point inward, but points outward by default
	normals = [-lsqnormal(el, **nargs) for el in elements]

	# Figure the propagation angles
	thetas = [np.arccos(np.dot(dirs, ne))
			for dirs, ne in izip(directions, normals)]

	return distances, thetas


def sigwidths(datfiles, chans, reffile, fs, thetas, dists, freqs, queue=None, **kwargs):
	'''
	Call sigffts() followed by dirwidths() on the output. Arguments are
	passed to their respective routines. The kwargs are passed to sigffts()
	to supply optional arguments.

	Returns zip(chans, out), where out is the output of dirwidths(). If
	queue is not None, the result will be passed to queue.put() before
	return.
	'''
	sfts, df = sigffts(datfiles, chans, reffile, fs, **kwargs)
	widths = dirwidths(sfts, thetas, dists, freqs, df)
	widthpairs = zip(chans, widths)

	# Try to put the result on the queue
	try: queue.put(widthpairs)
	except AttributeError: pass

	return widthpairs


def sigffts(datfiles, chans, reffile, fs, refwin=None, osamp=1, nsamp=None):
	'''
	For each habis.formats.WaveformSet file in the iterable datfiles, pull
	round-trip Waveform objects for all channels in the sequence chans,
	align the waveform with the Waveform object stored in reffile,
	optionally window the resulting waveform with a window characterized by
	the (start, length, [tailwidth]) sequence refwin (if refwin is None,
	windowing is skipped), and compute the DFT with Waveform.fft().
	
	The optional osamp will be passed to Waveform.aligned() when aligning
	waveforms to the reference. If nsamp is not None, all waveforms are
	truncated or zero-padded to a length nsamp. Otherwise, nsamp is assumed
	to be the largest of the nsamp parameters for all WaveformSet objects
	in datfiles.

	The window parameters (start, length) are passed directly to
	Waveform.window, while the optional tailwidth will be used to construct
	a Hann window as np.hanning(2 * tailwidth) to be passed as the tails
	argument.

	Returned is an ndarray of shape (len(chans), len(datfiles), nfsamp),
	where nfsamp is the number of samples necessary to store the output of
	the above-described Waveform.fft() operation; along with the common bin
	width, df = fs / nsamp, for the DFTs.
	'''
	# Open the data and reference files
	wsets = [WaveformSet.fromfile(dfile) for dfile in datfiles]
	ref = Waveform.fromfile(reffile)

	# Build window tails, if appropriate
	try: tails = np.hanning(2 * refwin[2])
	except (TypeError, IndexError): tails = None

	# Figure the output shape and transform type
	if nsamp is None: nsamp = max(wset.nsamp for wset in wsets)
	dtype = np.find_common_type([wset.dtype for wset in wsets], [])
	isReal = not np.issubdtype(dtype, np.complexfloating)
	nfsamp = (int(nsamp // 2) + 1) if isReal else nsamp

	# Create a place to store aligned waveforms
	sdat = np.empty((len(chans), len(wsets), nfsamp), dtype=np.complex128)

	for i, ch in enumerate(chans):
		for j, wset in enumerate(wsets):
			# Align each waveform with the reference
			sig = wset[ch,ch].aligned(ref, osamp=osamp)
			# Window the waveform if desired
			if refwin is not None:
				sig = sig.window(refwin[:2], tails=tails)
			# Compute and store the DFT
			sdat[i,j,:] = sig.fft((0, nsamp), isReal)

	return sdat, fs / nsamp


def dirwidths(phis, thetas, dists, freqs, df):
	'''
	Given an ndarray phis of shape (Nc, Nr, Nf) that stores, as in the
	output of sigffts(), the DFTs of backscatter measurements from Nc
	channels to Nr along the last axis, each with a corresponding
	propagation angle in thetas and an element-to-surface distance in
	dists, both ndarrays of shape (Nc, Nr), determine the width parameter A
	that best predicts (in the least squares sense) a directivity pattern
	of the form 

		D(w, sin(theta)) = exp(-2 * A * sin(theta) * w**2)

	in the spectral domain, where w is the radian frequency.

	The sequence freqs specifies a list of POSITIVE DFT frequency bins that
	will be used in the least-squares solution. Each frequency bin freqs[i]
	has a radian frequency w[i] = 2 * pi * df * freqs[i].
	'''
	newax = np.newaxis
	phis = np.asarray(phis)
	nchan, nrefl, nfsamp = phis.shape

	dists = np.asarray(dists)
	thetas = np.asarray(thetas)

	freqs = np.asarray(freqs)
	nfreqs = len(freqs)

	# Correct for distances
	phis = phis * (dists[:,:,newax] / np.max(dists))**2

	# Compute log-magnitude ratios to isolate directivity factor
	# Pull out frequencies of interest
	laphis = np.log(np.abs(phis[:,:,freqs]))
	y = laphis[:,:,newax,:] - laphis[:,newax,:,:]

	# Compute the angular difference terms in the directivity exponent
	sth = np.sin(thetas)
	x = (2 * (sth[:,newax,:,newax] - sth[:,:,newax,newax]) * 
			(2 * np.pi * df * freqs[newax,newax,newax,:])**2)

	# Compute the per-channel least-squares solutions
	xtx = np.sum(x * x, axis=(1,2,3))
	xty = np.sum(x * y, axis=(1,2,3))
	return xty / xtx


def equalizerEngine(config):
	'''
	Use the MultiPointTrilateration and PlaneTrilateration classes in
	habis.trilateration to determine, iteratively from a set of
	measurements of round-trip arrival times, the unknown positions of
	a set of reflectors followed by estimates of the positions of the
	hemisphere elements.
	'''
	esec = 'equalizer'
	msec = 'measurement'
	ssec = 'sampling'
	try:
		# Try to grab the input waveform sets
		datafiles = config.getlist(esec, 'waveset')
		if len(datafiles) < 1:
			err = 'Key waveset must contain at least one entry'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify waveset in [%s]' % esec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the reference waveform
		reffile = config.get(msec, 'reference')
	except Exception as e:
		err = 'Configuration must specify reference in [%s]' % msec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the reflector positions
		rflfile = config.get(esec, 'reflectors')
	except Exception as e:
		err = 'Configuration must specify reflectors in [%s]' % esec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the element positions by facet
		eltfiles = config.getlist(esec, 'elements')
		if len(eltfiles) < 1:
			err = 'Key elements must contain at least one entry'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Configuration must specify elements in [%s]' % esec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the reflector radius
		radius = config.getfloat(msec, 'radius')
	except Exception as e:
		err = 'Configuration must specify radius in [%s]' % msec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the output file
		outfile = config.get(esec, 'output')
	except Exception as e:
		err = 'Configuration must specify output in [%s]' % esec
		raise HabisConfigError.fromException(err, e)

	try:
		# Determine the sampling frequency
		per = config.getfloat(ssec, 'period')
		fs = 1. / per
	except Exception as e:
		err = 'Configuration must specify period in [%s]' % ssec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the DFT bins to use for the width optimization
		freqs = config.getrange(esec, 'freqs')
	except Exception as e:
		err = 'Configuration must specify freqs in [%s]' % esec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the number of processes to use (optional)
		nproc = config.getint('general', 'nproc',
				failfunc=process.preferred_process_count)
	except Exception as e:
		err = 'Invalid specification of optional nproc in [general]'
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the oversampling rate and sample count (optional)
		osamp = config.getint(ssec, 'osamp', failfunc=lambda: 1)
		nsamp = config.getint(ssec, 'nsamp', failfunc=lambda: None)
	except Exception as e:
		err = 'Invalid specification of optional osamp or nsamp in [%s]' % ssec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the reference window (optional)
		refwin = config.getlist(esec, 'refwin',
				mapper=int, failfunc=lambda: None)
		if refwin and (len(refwin) < 2 or len(refwin) > 3):
			err = 'Window does not specify appropriate parameters'
			raise HabisConfigError(err)
	except Exception as e:
		err = 'Invalid specification of optional window in [%s]' % esec
		raise HabisConfigError.fromException(err, e)

	try:
		# Grab the channels on which to pull waveforms
		channels = config.getrange(esec, 'channels', failfunc=lambda: None)
	except Exception as e:
		err = 'Invalid specification of optional channels in [%s]' % esec
		raise HabisConfigError.fromException(err, e)

	# Load the element and reflector positions, then compute distances and angles
	elements = [np.loadtxt(efile) for efile in eltfiles]
	nedim = elements[0].shape[1]
	for el in elements[1:]:
		if el.shape[1] != nedim:
			raise ValueError('Dimensionality of all element files must agree')
	# Ignore an optional sound-speed column in the reflector coordinates
	reflectors = np.loadtxt(rflfile)[:,:nedim]
	distances, thetas = wavepaths(elements, reflectors)

	# Concatenate the element, distance, and angle lists
	elements = np.concatenate(elements, axis=0)
	distances = np.concatenate(distances, axis=0)
	thetas = np.concatenate(thetas, axis=0)

	# Build the channel list of the default is not provided
	nchan = elements.shape[0]
	if channels is None:
		channels = range(nchan)
	elif len(channels) != nchan:
		raise ValueError('Number of element coordinates must match configured channel list')

	# Create a result queue
	queue = multiprocessing.Queue(nproc)

	with process.ProcessPool() as pool:
		kwargs = dict(refwin=refwin, osamp=osamp, nsamp=nsamp, queue=queue)
		for i in range(nproc):
			# Build the argument list with unique channel indices
			args = (datafiles, channels[i::nproc], reffile, fs, 
					thetas[i::nproc,:], distances[i::nproc,:], freqs)
			procname = process.procname(i)
			pool.addtask(target=sigwidths, name=procname, args=args, kwargs=kwargs)

		pool.start()

		# Wait for all processes to respond
		responses = 0
		widths = []
		while responses < nproc:
			try:
				widths.extend(queue.get(timeout=0.1))
				responses += 1
			except Queue.Empty: pass

		pool.wait()

	widths.sort()
	np.savetxt(outfile, [w[1] for w in widths], fmt='%16.8f')


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Read the configuration file
	try:
		config = HabisConfigParser.fromfile(sys.argv[1])
	except:
		print >> sys.stderr, 'ERROR: could not load configuration file %s' % sys.argv[1]
		usage(sys.argv[0])
		sys.exit(1)

	# Call the calculation engine
	equalizerEngine(config)
