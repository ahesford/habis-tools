'''
Routines for manipulating HABIS data file formats.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import mmap
import numpy as np
import re, os
import pandas
import struct

from collections import OrderedDict

def findenumfiles(dir, prefix='.*?', suffix='', ngroups=1):
	'''
	Find all files in the directory dir with a name matching the regexp
	r'^<PREFIX>(-([0-9]+)){ngroups}<SUFFIX>$', where <PREFIX> is replaced
	with an optional prefix and <SUFFIX> is replaced with an optional
	suffix to restrict the search, and return a list of tuples in which the
	first item is the name and subsequent entries are the matched integers
	(which will number ngroups) in left-to-right order.
	'''
	from os.path import join

	if ngroups < 1:
		raise ValueError('At least one number group must be specified')

	# Build the number-matching portion
	numstr = '-([0-9]+)' * ngroups
	# Enumerate the matching groups (0 is the whole matching string)
	grpidx = tuple(range(ngroups + 1))
	# Build the regexp and filter the list of files in the directory
	regexp = re.compile(r'^%s%s%s$' % (prefix, numstr, suffix))
	# When converting matched groups to integers, discard the whole-string group
	return [tuple([join(dir, f)] + [int(g) for g in m.group(*grpidx)[1:]])
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
	# Numpy record describing a receive-channel waveform record
	rechdr = np.dtype([('idx', '<u4'), ('crd', '<3f4'), ('win', '<2u4')])
	# struct format string for WaveformSet file header format
	hdrfmt = '<4s2I2s4I'
	# A bidirectional mapping between typecodes and Numpy dtype names
	from pycwp.util import bidict
	typecodes = bidict(I2 = 'int16', I4 = 'int32', I8 = 'int64',
			F2 = 'float16', F4 = 'float32', F8 = 'float64',
			C4 = 'complex64', C8 = 'complex128')

	@staticmethod
	def _verify_file_version(version):
		'''
		Ensure that the provided version matches one supported by the
		class. If not, raise a ValueError. If so, just return the
		version tuple.
		'''
		major, minor = version
		if major != 1: raise ValueError('Major version must be 1')
		if 0 != minor != 1: raise ValueError('Minor version must be 0 or 1')
		return major, minor


	@classmethod
	def makercvhdr(cls, idx, crd, win):
		'''
		From the channel index idx, the 3-element sequence crd, and the
		2-element sequence win, create a single Numpy record of the
		class's rechdr that contains the values.
		'''
		# Create an empty record
		hdr = np.zeros((1,), dtype=cls.rechdr)[0]
		hdr['idx'] = idx
		hdr['crd'][:] = crd
		hdr['win'][:] = win
		return hdr


	@classmethod
	def fromfile(cls, f, *args, **kwargs):
		'''
		Create a new WaveformSet object and use load() to populate the
		object with the contents of the specified file (a file-like
		object or a string naming the file).

		Extra args and kwargs are passed to the load() method.
		'''
		# Create an empty set, then populate from the file
		wset = cls([])
		wset.load(f, *args, **kwargs)
		return wset


	@classmethod
	def fromwaveform(cls, wave, copy=False):
		'''
		Create a new WaveformSet object with a single transmit index
		and a single receive index (both 0) with a sample count and
		data type defined by the provided Waveform wave. The sole
		waveform record will be populated with wave.

		If copy is False, the record in the WaveformSet will, whenever
		possible, capture a reference to the waveform data instead of
		making a copy. If copy is True, a copy will always be made.

		The f2c parameter of the WaveformSet will be 0.

		The 'crd' record of the waveform record will be [0., 0., 0.].
		'''
		# Create the set
		wset = cls([0], wave.nsamp, 0, wave.dtype)
		# Create the sole record; this grows the 1-D signal to 2-D
		hdr = cls.makercvhdr(0, [0., 0., 0.], wave.datawin)
		wset.setrecord(hdr, wave.getsignal(wave.datawin), copy)
		return wset


	@classmethod
	def packfilehdr(cls, dtype, nchans, nsamp, f2c=0, ver=(1,0)):
		'''
		Pack the Numpy dtype, built-in scalar type object, or type
		name; a channel count nchans = (nrx, ntx); an acquisition
		sample count nsamp; and a fire-to-capture delay f2c into a
		header string for a waveform-set file.

		The version number may be specifed as ver = (major, minor).
		'''
		major, minor = ver
		# Grab the name of the Numpy dtype
		typename = np.dtype(dtype).name
		fmtstr = cls.typecodes.inverse[typename][0]
		nrx, ntx = nchans
		# Encode the header information
		hdr = struct.pack(cls.hdrfmt, 'WAVE', major, minor, 
				fmtstr, f2c, nsamp, nrx, ntx)
		return hdr


	@classmethod
	def unpackfilehdr(cls, header):
		'''
		Unpack the packed waveform set file header into a return tuple:

			dtype, Numpy dtype of waveform records
			nchans, tuple defining channel counts (nrx, ntx)
			f2c, fire-to-capture delay in 20-MHz samples
			nsamp, total number of 20-MHz samples captured
			version, tuple indicating file version (major, minor)

		The provided header may be a file-like object or a string. If
		header is a string, it must have a length exactly equal to
		struct.calcsize(cls.hdrfmt). If header is a file-like object,
		enough bytes will by read() from the current position to parse
		the header fields. Bytes after the header will be ignored.
		'''
		# Try to read the bytes, or pull them from the provided string
		hdrbytes = struct.calcsize(cls.hdrfmt)
		try: hstr = header.read(hdrbytes)
		except AttributeError: hstr = header

		if len(hstr) != hdrbytes: raise ValueError('Inappropriate header length')

		unpacked = struct.unpack(cls.hdrfmt, hstr)

		if unpacked[0] != 'WAVE':
			raise ValueError('Header string does not contain WAVE magic number')

		# Copy the version tuple
		version = unpacked[1:3]
		# Decode the datatype
		dtype = np.dtype(cls.typecodes[unpacked[3]])
		# Copy the fire-to-capture delay, sample count, and channel counts
		f2c, nsamp = unpacked[4:6]
		nchans = unpacked[6:8]

		return dtype, nchans, f2c, nsamp, version


	@staticmethod
	def packtxlist(txidx):
		'''
		Encode the provided transmit-index list into bytes suitable for
		writing after a global file header in WAVE version (1,0) files.
		'''
		# Ensure the input is a 1-D list of uints
		txidx = np.asarray(txidx, dtype=np.uint32)
		if txidx.ndim != 1:
			raise ValueError('Transmit-index list must be a 1-D list')

		# In the header, indices are one-based
		txhdr = (txidx + np.array(1, dtype=np.uint32)).tobytes()
		return txhdr


	@staticmethod
	def unpacktxlist(f, ntx):
		'''
		Decode a list of ntx transmit indices from the object f, which
		is either a file-like object or a string. If f is a file-like
		object, the number of bytes necessary to encode the list is
		read() from the current position of f. Otherwise, f must be a
		string of exactly the number of bytes necessary to store the
		list.
		'''
		txbytes = ntx * np.dtype('uint32').itemsize
		try: txstr = f.read(txbytes)
		except AttributeError: txstr = f

		if len(txstr) != txbytes:
			raise ValueError('Could not read encoded transmit-index list')

		# In the header, indices are one-based
		txidx = (np.fromstring(txstr, dtype=np.uint32)
				- np.array(1, dtype=np.uint32))

		if len(txidx) != ntx:
			raise ValueError('Parsed transmit-index list is not the proper size')

		return txidx


	@classmethod
	def empty_like(cls, wset):
		'''
		Create a new instance of WaveformSet configured exactly as
		wset, except without any waveform records.
		'''
		return cls(wset.txidx, wset.nsamp, wset.f2c, wset.dtype)


	def __init__(self, txchans, nsamp=4096, f2c=0, dtype=np.dtype('int16')):
		'''
		Create an empty WaveformSet object corresponding acquisitions
		of a set of waveforms from the (0-based) transmission indices
		in txchans, acquired with a total of nsamp samples per
		acquisition after a fire-to-capture delay of f2c samples.
		Waveform arrays are stored with the specified Numpy dtype.
		'''
		if f2c < 0:
			raise ValueError('Fire-to-capture delay f2c must be nonnegative')
		if nsamp < 0:
			raise ValueError('Number of samples nsamp must be nonnegative')

		# Record the waveform dtype (a read-only property)
		self._dtype = np.dtype(dtype)
		# Record the total number of acquired samples (setting is OK,
		# but the new value is checked for consistence with records)
		self._nsamp = nsamp
		# The fire-to-capture delay can be set with impunity
		self.f2c = f2c

		# Create an empty record dictionary and tx index map
		# OrderedDict is used so the keys remain in file order
		self._records = OrderedDict()
		self.txidx = txchans


	def encodefilehdr(self, dtype=None, ver=None):
		'''
		Return a byte string, suitable for writing to a file, that
		represents the file-level header for this WaveformSet. The
		record data type can be overridden with the dtype argument. The
		header version number can be specified as ver=(major, minor)
		or, if None, defaults to (1,1) when self.txidx is equivalent to
		range(self.ntx), and (1,0) otherwise.

		Useful for creating an appendable WaveformSet output file that
		can be written in parallel.
		'''
		if dtype is None: dtype = self.dtype
		nchans = (self.nrx, self.ntx)

		if ver is None:
			ver = (1,1)
			last = -1
			for tid in self.txidx:
				if tid != last + 1:
					ver = (1,0)
					break
				last = tid

		major, minor = self._verify_file_version(ver)

		# Create the header bytes
		hdr = type(self).packfilehdr(dtype, nchans, self.nsamp, self.f2c, ver)
		# For version (1,0) files, explicitly encode the txlist
		if minor == 0: hdr += type(self).packtxlist(self.txidx)
		return hdr


	def store(self, f, append=False, ver=(1,0)):
		'''
		Write the WaveformSet object to the data file in f (either a
		name or a file-like object that allows writing).

		If append is True, an unopened file is opened for appends
		instead of cloberring an existing file. In this case, writing
		of the file-level header is skipped. It is the caller's
		responsibility to assure that an existing file header is
		consistent with records written by this method in append mode.

		** NOTE **
		Because the WaveformSet may map some input file for waveform
		arrays after calling load(), calling store() with the same file
		used to load() may cause unexpected behavior.
		'''
		# Open the file if it is not open
		if isinstance(f, basestring):
			f = open(f, mode=('wb' if not append else 'ab'))

		if not append:
			# Write the file-level header
			f.write(self.encodefilehdr())

		# Write each record in turn
		for idx, (hdr, waveforms) in self._records.iteritems():
			# The file uses 1-based indices for channel indices
			hdrc = hdr.copy()
			hdrc['idx'] += 1
			hdrc.tofile(f)
			waveforms.tofile(f)


	def load(self, f, ver=None):
		'''
		Associate the WaveformSet object with the data in f, a
		file-like object or string specifying a file name. If f is a
		file-like object, parsing starts from the current file
		position.

		Existing waveform records will be eliminated. All parameters of
		the WaveformSet (arguments to the constructor) will be reset to
		values specified in the file header.

		Each block of waveform data is memory-mapped from the source
		file. This mapping is copy-on-write; changes do not persist.

		If ver is not None, it must be a (major, minor) that will
		override the version number stored in the file header.

		** NOTE **
		In the file, each receive channel has an associated transmit
		index. The transmit index is 1-based in the file, but is
		converted to a 0-based record in this encapsulation.
		'''
		# Open the file if it is not open
		if isinstance(f, basestring):
			f = open(f, mode='rb')

		# Parse the file header
		dtype, (nrx, ntx), f2c, nsamp, fver = type(self).unpackfilehdr(f)

		# Override the file version if necessary
		if ver is None: ver = fver

		major, minor = self._verify_file_version(ver)

		if nsamp < 0:
			raise ValueError('Number of samples in acquisition window must be nonnegative')
		if f2c < 0:
			raise ValueError('Fire-to-capture delay must be nonnegative')

		if minor == 0:
			# Read the transmit list from the file
			self.txidx = type(self).unpacktxlist(f, ntx)
		elif minor == 1:
			# The transmit list is implicit
			self.txidx = range(ntx)

		# Clear the record array
		self.clearall()
		# Set the data type sample count, and fire-to-capture delay
		self.dtype = dtype
		self.nsamp = nsamp
		self.f2c = f2c

		# Record the start of the waveform data records in the file
		fpos = f.tell()

		# Use a single Python mmap buffer for backing data
		# This avoids multiple openings of the file for independent maps
		buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
		f.seek(fpos)

		# Parse through the specified number of receive records
		for ridx in range(nrx):
			# If a header cannot be read, processing stops
			try: hdr = np.fromfile(f, count=1, dtype=self.rechdr)[0]
			except IndexError: break

			# Convert transmit index to 0-based
			hdr['idx'] -= 1
			# Determine the shape of the waveform
			waveshape = (ntx, hdr['win'][1])
			# Determine the start and end of waveform data block
			fstart = f.tell()
			# Return a view of the map
			wavemap = np.ndarray(waveshape, dtype=dtype,
					buffer=buf, order='C', offset=fstart)
			# Add the record to the set
			self.setrecord(hdr, wavemap, copy=False)
			# Skip to the next header
			f.seek(fstart + wavemap.nbytes)


	@property
	def rxidx(self):
		'''
		Return a list of receive-channel indices in file order.
		'''
		return self._records.keys()


	@property
	def txidx(self):
		'''
		Return a list of transmit-channel indices in file order.
		'''
		return self._txmap.keys()


	@txidx.setter
	def txidx(self, txidx):
		'''
		Set the mapping between transmit indices and file-order
		waveform indices.
		'''
		# If receive records exist, ensure that the transmit counts
		# for each record match the count of the provided index list
		ntx = len(txidx)
		for hdr, waves in self.allrecords():
			if waves.shape[0] != ntx:
				raise ValueError('Count of specified transmit indices does not match existing records')

		self._txmap = OrderedDict((int(tx), i) for i, tx in enumerate(txidx))


	@property
	def ntx(self):
		'''
		Return the number of transmissions per receive channel.
		'''
		return len(self._txmap)


	@property
	def nrx(self):
		'''
		Return the number of receive channels in this waveform set.
		'''
		return len(self._records)


	@property
	def dtype(self):
		'''
		Return the datatype used to store waveforms.
		'''
		return self._dtype


	@dtype.setter
	def dtype(self, value):
		if self.nrx > 0:
			raise ValueError('Cannot change datatype with existing records')
		self._dtype = np.dtype(value)


	@property
	def nsamp(self):
		'''
		Return the total number of samples collected in the acquisitions.
		'''
		return self._nsamp


	@nsamp.setter
	def nsamp(self, newval):
		'''
		Set the total number of samples in the acquisition window.
		Ensure existing records don't fall outside of the window.
		'''
		# Force the new value to be an integer
		nsamp = int(newval)
		# Check all existing records to ensure their windows don't
		# extend past the new acquisition window
		for hdr, wforms in self.allrecords():
			start, length = hdr['win']
			if start + length > nsamp:
				raise ValueError('Acquisition window fails to contain stored waveforms')

		# Set the new value
		self._nsamp = nsamp


	def tx2row(self, tid):
		'''
		Convert a transmit-channel index into a waveform-array row index.
		'''
		return self._txmap[tid]


	def row2tx(self, row):
		'''
		Convert a waveform-array row index to a transmit-channel index.
		'''
		# Use the ordered keys in the txmap to pull out the desired row
		return self._txmap.keys()[row]


	def getrecord(self, rid, tid=None, window=None, dtype=None):
		'''
		Return a (header, waveforms) record for the receive channel
		with channel index rid. If window is None and dtype is None,
		the waveforms data array is a view of the internal
		copy-on-write memory map.

		If tid is not None, it should be a scalar integer or an
		iterable of integers that represent transmit channel indices to
		pull from the waveform array. When tid is a scalar, a 1-D array
		is returned to represent the samples for the specified
		transmission. When tid is an iterable (even of length 1), a 2-D
		array is returned with transmit indices along the rows (in the
		order specified by tid) and waveform samples along the columns.
		When tid is None, self.txidx is assumed.

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
		if tid is None:
			# With no tid, pull all transmissions
			tcidx = range(self.ntx)
		else:
			try:
				# Map the transmit IDs to row indices
				tcidx = [self.tx2row(t) for t in tid]
				singletx = False
			except TypeError:
				# Handle mapping for a scalar tid
				tcidx = self.tx2row(tid)
				singletx = True

		# Grab receive record, copy header to avoid corruption
		hdr, waveforms = self._records[rid]
		hdr = hdr.copy()

		# If the data is not changed, just return a view of the waveforms
		if window is None and dtype is None:
			return hdr, waveforms[tcidx,:]

		# Pull an unspecified output window from the header
		if window is None:
			window = hdr['win']
		# Pull an unspecified data type from the waveforms
		if dtype is None or dtype == 0:
			dtype = waveforms.dtype

		# Create an output array to store the results
		oshape = (1 if singletx else len(tcidx), window[1])
		output = np.zeros(oshape, dtype=dtype)

		try:
			# Figure out the overlapping sample window
			# Raises TypeError if overlap() returns None
			from pycwp.cutil import overlap
			ostart, istart, wlen = overlap(window, hdr['win'])
			oend, iend = ostart + wlen, istart + wlen

			# Copy portion of waveforms overlapping the window
			output[:,ostart:oend] = waveforms[tcidx,istart:iend]
		except TypeError: pass

		# For a scalar tid, collapse the 2-D array
		if singletx: output = output[0]

		# Override the window in the header copy
		hdr['win'][:] = window

		return hdr, output


	def getwaveform(self, rid, tid, *args, **kwargs):
		'''
		Return, as one or more habis.sigtools.Waveform objects, the
		waveform(s) recorded at receive-channel index rid from the
		(scalar or iterable of) transmission(s) tid.

		If tid is a scalar, a single Waveform object is returned.
		Otherwise, is tid is an iterable or None (which pulls all
		transmissions), a list of Waveform objects is returned.

		Extra args and kwargs are passed through to getrecord().
		'''
		from .sigtools import Waveform
		# Grab the relevant row of the record
		hdr, wform = self.getrecord(rid, tid, *args, **kwargs)

		# Wrap a single desired signal in a Waveform object
		if np.ndim(wform) == 1:
			return Waveform(self.nsamp, wform, hdr['win'][0])
		else:
			return [Waveform(self.nsamp, w, hdr['win'][0]) for w in wform]


	def __getitem__(self, key):
		'''
		A convenience method to call getwaveform(rid, tid) for provided
		receive and transmit indices. Only a key of the form (rid, tid)
		is supported.
		'''
		try:
			# Split the indices
			rid, tid = key
		except (TypeError, ValueError):
			raise TypeError('Item key should be a sequence of two integer values')

		if int(rid) != rid or int(tid) != tid:
			raise TypeError('Waveform indices must be integer-compatible')

		return self.getwaveform(rid, tid)


	def delrecord(self, rid):
		'''
		Delete the waveform record for the receive-channel index rid.
		'''
		del self._records[rid]


	def clearall(self):
		'''
		Delete all waveform records in the set.
		'''
		# Just create a new record dictionary
		self._records = OrderedDict()


	def setrecord(self, hdr, waveforms=None, copy=True):
		'''
		Save a waveform record consisting of the provided header and
		waveform array. If a record for the receive channel specified
		in the header already exists, it will be overwritten.
		Otherwise, the record will be created.

		The waveform array must either be a Numpy ndarray or None. When
		waveforms takes the special value None, a new, all-zero
		waveform array is created (regardless of the value of copy).

		If copy is False, a the record will store a reference to the
		waveform array if the types are compatible. If copy is True, a
		local copy of the waveform array, cast to this set's dtype,
		will always be made.
		'''
		# Check that the header bounds make sense
		if hdr['win'][0] + hdr['win'][1] > self.nsamp:
			raise ValueError('Waveform sample window exceeds acquisition window duration')

		if int(hdr['idx']) != hdr['idx']:
			raise TypeError('Channel index must be integer-compatible')


		if waveforms is None:
			# Create an all-zero waveform array
			wshape = (self.ntx, hdr['win'][1])
			waveforms = np.zeros(wshape, dtype=self.dtype)
		else:
			try:
				if copy or waveforms.dtype != self.dtype:
					# Make a copy of the waveform in proper format
					raise TypeError('Conversion of dtypes required')
			except (AttributeError, TypeError):
				waveforms = np.array(waveforms, dtype=self.dtype)

			# Pad 0-d and 1-d waveforms to 2-d
			if waveforms.ndim < 2:
				waveforms = waveforms[[None] * (2 - waveforms.ndim)]

			# Check the proper shape of the provided array
			ntx, nsamp = waveforms.shape
			if ntx != self.ntx:
				raise ValueError('Waveform array does not match transmission count for set')
			if nsamp != hdr['win'][1]:
				raise ValueError('Waveform array does not match sample count specified in header')

		# Add or replace the record
		self._records[int(hdr['idx'])] = (hdr, waveforms)


	def setwaveform(self, rid, tid, wave):
		'''
		Replace the waveform at receive index rid and transmit index
		tid with the provided habis.sigtools.Waveform wave. When replacing
		the existing waveform, the signal will be padded and clipped as
		necessary to fit into the window defined for the record.
		'''
		tcidx = self.tx2row(tid)

		# Pull the existing record
		hdr, wfrec = self._records[rid]

		# Overwrite the transmit row with the input waveform
		wfrec[tcidx,:] = wave.getsignal(window=hdr['win'], dtype=wfrec.dtype)


	def allrecords(self, *args, **kwargs):
		'''
		Return a generator that fetches each record, in channel-index
		order, using self.getrecord(rid, window, dtype).
		'''
		for rid in sorted(self.rxidx):
			yield self.getrecord(rid, *args, **kwargs)
