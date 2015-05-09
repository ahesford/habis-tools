'''
Routines for manipulating HABIS data file formats.
'''

import mmap
import numpy as np
import re, os
import pandas

from itertools import count as icount
from pycwp import cutil

def findenumfiles(dir, prefix='.*?', suffix=''):
	'''
	Find all files in the directory dir with a name matching the regexp
	r'^<PREFIX>([0-9]+)<SUFFIX>$', where <PREFIX> is replaced with an
	optional prefix and <SUFFIX> is replaced with an optional suffix to
	restrict the search, and return a list of tuples in which the first
	item is the name and the second item is the matched integer.
	'''
	# Build the regexp and filter the list of files in the directory
	regexp = re.compile(r'^%s([0-9]+)%s$' % (prefix, suffix))
	return [(os.path.join(dir, f), int(m.group(1)))
			for f in os.listdir(dir) for m in [regexp.match(f)] if m]


def specreptype():
	'''
	Returns a numpy data type consisting of a 64-bit complex component,
	labeled 'val', which stores the magnitude of a spectral component and a
	64-bit integer, labeled 'idx', which stores the component's FFT index.
	'''
	return np.dtype([('val', np.complex64), ('idx', np.int64)])


def splitspecreps(a):
	'''
	Break a record array a of concatenated spectral representations, with
	dtype habis.formats.specreptype(), into a list of record arrays
	corresponding to each group of spectral representations in the original
	array. The number of records in the first group (output[0]) is
	specified by n[0] = (a[0]['idx'] + 1), with output[0] = a[:n[0]].

	The number of records in a subsequent group (output[i]) is given by

		n[i] = (a[sum(n[:i-1])]['idx'] + 1),

	with output[i] = a[sum(n[:i-1]):sum(n[:i])].
	'''
	start = 0
	output = []
	while start < len(a):
		nvals = a[start]['idx'] + 1
		if nvals < 1: raise ValueError('Spectral representation counts must be positive')
		grp = a[start:start+nvals]
		if len(grp) < nvals: raise ValueError('Could not read specified number of records')
		output.append(a[start:start+nvals])
		start += nvals
	return output


def countspecreps(f):
	'''
	For a file f that contains sequence of spectral representations, return
	the number of components in each group within the sequence. Thus, if A
	represents the array of habis.formats.specreptype() records listed in the
	file f, the output array n will have

		n[0] = (A[0]['idx'] + 1), and
		n[i] = (A[sum(n[:i-1])]['idx'] + 1).
	'''
	dtype = specreptype()
	# Open the file and determine its size
	infile = open(f, 'rb')
	infile.seek(0, os.SEEK_END)
	fend = infile.tell()
	infile.seek(0, os.SEEK_SET)
	# Scan through the file to pick up all of the counts
	n = []
	while (infile.tell() < fend):
		# Read the header record and add it to the list
		nrec = np.fromfile(infile, dtype=dtype, count=1)[0]['idx']
		n.append(nrec + 1)
		# Skip over the records for this group
		infile.seek(nrec * dtype.itemsize, os.SEEK_CUR)

	return n


def repreducer(n):
	'''
	This is a factory function that returns a reducer function, suitable
	for use in readfiresequence and readfirecapture, which selects only
	rows whose repetition index matches the specified integer n.
	'''
	def reducefunc(mat): return mat[mat[:,1].astype(int) == n]
	return reducefunc


def readfirecapture(f, reducer=None):
	'''
	Read the capture of a single HABIS fire sequence (with any number of
	transmit repetitions) in CSV format. The file has 4 header lines and is
	comma-delimited. The format of each line is a sequence of integers

		channel, repetition, samples...

	where samples are in the range [-8192,8192). Channel values are indexed
	from zero.

	The data is sorted first by channel and then by repetition index before
	processing.

	The return value is a tuple (output, channels, repetitions), where
	output is 3-D array of the form output[i,j,k], where i is the receive
	channel index, j is the repetition, and k is the sample index. Every
	receive channel must contain the same number of repetitions or a
	ValueError will be raised. The list channels contains elements that
	indicate the channel indices identified in the file, such that
	channels[i] is the listed channel index for slice output[i,:,:].
	The list repetitions is similarly defined such that reptitions[j] is
	the listed repetition index for slice output[:,j,:].

	If reducer is not None, it should be a callable that takes as input the
	raw array data read from f and returns a filtered version of the data
	that will be processed as that were the raw data read from the file.
	'''
	# Read the data and use the reducer filter if appropriate
	data = pandas.read_csv(f, skiprows=4, header=None).values
	# If reducer is None, a TypeError is raised; just ignore it
	try: data = reducer(data)
	except TypeError: pass

	# Sort the data according to channel and repetition
	idx = sorted((d[0], d[1], i) for i, d in enumerate(data[:,:2]))
	data = data[[v[-1] for v in idx]]
	# Count the channels and reptitions
	def counter(x, y):
		"Count the channel and repetition in a result dictionary tuple"
		try: x[0][y[0]] += 1
		except KeyError: x[0][y[0]] = 1
		try: x[1][y[1]] += 1
		except KeyError: x[1][y[1]] = 1
		return x
	channels, repetitions = reduce(counter, idx, ({}, {}))
	# Ensure that all channels have the same repetition count
	if len(set(channels.values())) != 1:
		raise ValueError('All channels must have the same number of reptitions')
	if len(set(repetitions.values())) != 1:
		raise ValueError('Each channel must have same set of reptition indices')

	# Strip out the channel and repetition indices
	channels = sorted(channels.keys())
	repetitions = sorted(repetitions.keys())

	nchan = len(channels)
	nreps = len(repetitions)
	nsamps = data.shape[-1] - 2

	return data[:,2:].reshape((nchan, nreps, nsamps)), channels, repetitions


def readfiresequence(fmt, findx, reducer=None):
	'''
	Read a series of HABIS fire capture fires whose names are given by the
	Python format string fmt. The string fmt is passed to the format
	function with each value in the sequence findx to produce a unique
	filename. The output arrays of readfirecapture() are collected, in
	sequence, and concatenated along a new first axis.

	The channel and reptition indices returned by readfirecapture() are
	ignored. However, because np.concatenate() is used to produce the
	concatenated output, every readfirecapture() array must have the same
	shape.

	The reducer is passed to readfirecapture for processing per-fire data.
	'''
	data = [readfirecapture(fmt.format(f), reducer=reducer)[0][np.newaxis,:,:,:]
			for f in findx]
	return np.concatenate(data, axis=0)


class WaveformSet(object):
	'''
	A class to encapsulate a (possibly multi-facet) set of pulse-echo
	measurements from a single target.
	'''
	# Create a type to describe a receive-channel record
	hdrtype = np.dtype([('idx', 'f4'), ('crd', '3f4'), ('win', '2i4')])

	@classmethod
	def makehdr(cls, idx, crd, win):
		'''
		From the channel index idx, the 3-element sequence crd, and the
		2-element sequence win, create a single Numpy record of the
		class's hdrtype that contains the values.
		'''
		# Create an empty record
		hdr = np.zeros((1,), dtype=cls.hdrtype)[0]
		hdr['idx'] = idx
		hdr['crd'][:] = crd
		hdr['win'][:] = win
		return hdr


	@classmethod
	def fromfile(cls, f, ntx=512, dtype=np.int16):
		'''
		Create a new WaveformSet object and use load() to populate the
		object with the contents of the specified file (a file-like
		object or a string naming the file).
		'''
		wset = cls(ntx, dtype)
		wset.load(f)
		return wset


	def __init__(self, ntx=512, dtype=np.int16):
		'''
		Create an empty WaveformSet object corresponding to ntx
		transmissions per receive channel, with waveform arrays stored
		in the specified Numpy dtype.
		'''
		# Record the number of transmissions and waveform datatype
		self._ntx = ntx
		self._dtype = dtype

		# Create an empty record array and tx/rx index maps
		self._records = []
		self._rxmap = {}
		# A dummy transmit-index map
		self._txmap = { i: i for i in range(ntx) }


	def store(self, f):
		'''
		Write the WaveformSet object to the data file in f (either a
		name or a file-like object that allows writing).

		** NOTE **
		Because the WaveformSet may map some input file for waveform
		arrays after calling load(), calling store() with the same file
		used to load() may cause unexpected behavior.
		'''
		# Open the file if it is not open
		if isinstance(f, (str, unicode)): f = open(f, mode='wb')

		# Write each record in turn
		for hdr, waveforms in self._records:
			# The file uses 1-based indices for channel indices
			hdrc = hdr.copy()
			hdrc['idx'] += 1
			hdrc.tofile(f)
			waveforms.tofile(f)


	def load(self, f, ntx=None, dtype=None):
		'''
		Associate the WaveformSet object with the data in f, a
		file-like object or string specifying a file name. If f is a
		file-like object, parsing starts from the current file
		position.
		
		Existing waveform records will be eliminated. Each receive
		channel in the data set is assumed to have records for ntx
		transmissions, stored with the specified Numpy datatype. If
		either of ntx or dtype is None, the current value associated
		with the class instance are used.

		Each block of waveform data is memory-mapped from the source
		file. This mapping is copy-on-write; changes do not persist.

		** NOTE **
		In the file, each receive channel has an associated transmit
		index. The transmit index is 1-based in the file, but is
		converted to a 0-based record in this encapsulation.
		'''
		# Open the file if it is not open
		if isinstance(f, (str, unicode)): f = open(f, mode='rb')

		# Clear the record array
		self._records = []

		# Set the data type and number of transmissions
		if dtype is not None:
			self._dtype = dtype
		else:
			dtype = self._dtype

		if ntx is not None: 
			self._ntx = ntx
		else:
			ntx = self._ntx

		# Use a single Python mmap buffer for backing data
		# This avoids multiple openings of the file for independent maps
		buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
		f.seek(0)

		# Parse through the file until the end is reached
		while True:
			# If a header cannot be read, processing stops
			try: hdr = np.fromfile(f, count=1, dtype=self.hdrtype)[0]
			except IndexError: break

			# Convert transmit index to 0-based
			hdr['idx'] -= 1
			# Determine the shape of the waveform
			waveshape = (ntx, hdr['win'][-1])
			# Determine the start and end of waveform data block
			fstart = f.tell()
			# Return a view of the map
			wavemap = np.ndarray(waveshape, dtype=dtype,
					buffer=buf, order='C', offset=fstart)
			self._records.append((hdr, wavemap))
			# Skip to the next header
			f.seek(fstart + ntx * hdr['win'][-1] * dtype().nbytes)
			
		# Create a map between transmit index and record number
		self._rxmap = { int(rec[0]['idx']): i for i, rec in enumerate(self._records) }
		# A dummy transmit-index map
		self._txmap = { i: i for i in range(ntx) }


	@property
	def rxidx(self):
		'''
		Return a list of receive-channel indices in arbitrary order.
		'''
		return self._rxmap.keys()


	@property
	def txidx(self):
		'''
		Return a list of transmit-channel indices in arbitrary order.
		'''
		return self._txmap.keys()


	@property
	def ntx(self):
		'''
		Return the number of transmissions per receive channel.
		'''
		return self._ntx


	@property
	def dtype(self):
		'''
		Return the datatype used to store waveforms.
		'''
		return self._dtype


	def __len__(self):
		'''
		Return the number of records in the data set.
		'''
		return len(self._records)


	def tx2row(self, tid):
		'''
		Convert a transmit-channel index into a row index in a waveform
		record array.
		'''
		try:
			return self._txmap[int(tid)]
		except KeyError:
			raise IndexError('The transmit index does not exist in this set')


	def rx2rec(self, rid):
		'''
		Convert a receive-channel index into a waveform record index
		according to file order.
		'''
		try:
			return self._rxmap[int(rid)]
		except KeyError:
			raise IndexError('The receive index does not exist in this set')


	def getrecord(self, rid, tid=None, window=None, dtype=None):
		'''
		Return a (header, waveforms) record for the receive channel
		with channel index rid. If window is None and dtype is None,
		the waveforms data array is a reference to the internal
		copy-on-write memory map.

		If tid is not None, it should be a channel index that specifies
		a specific transmission to pull from the waveforms. Otherwise,
		all transmit waveforms are returned.

		If window is not None, it should be a tuple (start, length)
		that specifies the first sample and length of the temporal
		window over which the waveforms are interpreted. Even if window
		matches the internal window in the header, a copy of the
		waveform array will be made.

		If dtype is not None, the output copy of the waveforms in the
		record will be cast to this datatype.

		If exactly one of window or dtype is None, the corresponding
		value from the record will be used.

		To force a copy without knowing or changing the window and
		dtype, pass dtype=0.
		'''
		# Convert the channel indices to file-order indices
		rcidx = self.rx2rec(rid)
		if tid is not None: tcidx = self.tx2row(tid)

		# Grab the record according to the transmit ID map
		header, waveforms = self._records[rcidx]
		# Make a copy of the header to avoid corruption
		header = header.copy()
		# If the data is not changed, just return a view of the waveforms
		if window is None and dtype is None:
			if tid is None:
				return header, waveforms
			else:
				return header, waveforms[tcidx,:]

		# Pull an unspecified output window from the header
		if window is None:
			window = header['win']
		# Pull an unspecified data type from the waveforms
		if dtype is None or dtype == 0:
			dtype = waveforms.dtype

		if tid is None: oshape = (waveforms.shape[0], window[1])
		else: oshape = (window[1],)

		# Create the output copy
		output = np.zeros(oshape, dtype=dtype)

		try:
			# Figure out the overlapping sample window
			# Raises TypeError if overlap() returns None
			ostart, istart, wlen = cutil.overlap(window, header['win'])
			oend, iend = ostart + wlen, istart + wlen

			# Copy the relevant portions of the waveforms
			if tid is None:
				output[:,ostart:oend] = waveforms[:,istart:iend]
			else:
				output[ostart:oend] = waveforms[tcidx,istart:iend]
		except TypeError: pass

		# Override the window in the header copy
		header['win'][:] = window

		return header, output


	def setrecord(self, hdr, waveforms):
		'''
		Save a waveform record consisting of the provided header and
		waveform array. If a record for the receive channel specified
		in the header already exists, it will be overwritten.
		Otherwise, the record array will be extended.

		The waveform array will be cast to the dtype indicated in this
		object instance.
		'''
		ntx, nsamp = waveforms.shape
		if ntx != self.ntx:
			raise ValueError('Waveform array does not match transmission count for set')
		if nsamp != hdr['win'][1]:
			raise ValueError('Waveform array does not match sample count specified in header')

		# Make a copy of the waveforms, converting to the right datatype
		wcrec = np.array(waveforms, dtype=self.dtype)
		# Pull the receive-channel index from the header
		rid = hdr['idx']

		try:
			# Replace an existing record
			rcidx = self.rx2rec(rid)
			self._records[rcidx] = (hdr, wcrec)
		except IndexError:
			# Add a new rx record
			self._rxmap[int(rid)] = len(self._records)
			self._records.append((hdr, wcrec))


	def setwaveform(self, rid, tid, waveform, window):
		'''
		Replace the waveform at receive index rid and transmit index
		tid with the provided waveform defined over the sample window
		as a 2-tuple (start, length) and 0 elsewhere. When replacing
		the existing waveform, the signal will be padded and clipped as
		necessary to fit into the window defined for the record.
		'''
		rcidx = self.rx2rec(rid)
		tcidx = self.tx2row(tid)

		# Pull the existing record
		hdr, wfrec = self._records[rid]
		# Determine the overlap of the input and output windows
		overlap = cutil.overlap(hdr['win'], window)

		if overlap is None:
			# Zero the waveform if there is no overlap
			wfrec[tcidx,:] = 0.
		else:
			# Copy the overlapping region and zero everywhere else
			ostart, istart, wlen = overlap
			oend, iend = ostart + wlen, istart + wlen
			wfrec[tcidx,ostart:oend] = waveform[istart:iend]
			wfrec[tcidx,:ostart] = 0.
			wfrec[tcidx,oend:] = 0.


	def allrecords(self, tid=None, window=None, dtype=None):
		'''
		Return a generator that fetches each record, in channel-index
		order, using self.getrecord(rid, tid, window, dtype).
		'''
		for rid in sorted(self.rxidx):
			yield self.getrecord(rid, tid, window, dtype)
