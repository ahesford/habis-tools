'''
Routines for manipulating HABIS data file formats.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import mmap
import numpy as np
import os
import struct

from itertools import repeat

from collections import OrderedDict, defaultdict, namedtuple
from functools import reduce

def _strict_int(x):
	ix = int(x)
	if ix != x:
		raise ValueError('Argument must be integer-compatible')
	return ix


def _strict_float(x):
	ix = float(x)


def _strict_nonnegative_int(x, positive=False):
	x = _strict_int(x)
	if positive and x <= 0:
		raise ValueError('Argument must be positive')
	elif x < 0:
		raise ValueError('Argument must be nonnegative')
	return x


def renderAndLoadYaml(data, **kwargs):
	'''
	Attempt to render the string data as a Mako template with kwargs passed
	to the Mako renderer with string_undefined=True.  Parse the rendered
	result as YAML using yaml.safe_load.

	If the Mako template engine cannot be imported, the data is parsed as
	pure YAML. Specifying kwargs when Mako cannot be imported raises a
	TypeError.
	'''
	from yaml import safe_load

	try:
		from mako.template import Template
	except ImportError:
		if kwargs:
			raise TypeError('Extra keyword arguments '
					'require Mako template engine')
		return safe_load(data)
	else:
		tmpl = Template(text=data, strict_undefined=True)
		return safe_load(tmpl.render(**kwargs))


def loadmatlist(files, *a, **k):
	'''
	A conveience function to produce the ordered dictionary

		OrderedDict(sorted(kv for f in files
				for kv in loadkeymat(f, *a, **k).iteritems()))

	If files is a string instead of any other iterable, it will be replaced
	with glob.glob(files) before being inserted into the above constructor.

	When files is a string, a special keyword argument, forcematch, may be
	provided. This argument will be stripped from the kwargs dictionary k
	and, when True, will cause an IOError to be raised if the glob matches
	no files. Otherwise, if forcematch is omitted or False, a glob that
	matches no files will cause an empty map to be returned.
	'''
	if isinstance(files, str):
		from glob import glob
		files = glob(files)
		forcematch = k.pop('forcematch', False)
		if forcematch and not files: raise IOError('No matches for glob "files"')

	return OrderedDict(sorted(kv for f in files
		for kv in loadkeymat(f, *a, **k).items()))


def loadkeymat(f, scalar=None, dtype=None, nkeys=None):
	'''
	A convenience function that will attempt to load a mapping from f using
	loadz_keymat or (if loadz_keymat fails) loadtxt_keymat. The optional
	arguments scalar and dtype, if not None, are passed as kwargs to either
	load function.

	If nkeys is not None, it will be used to verify the cardinality of keys
	in a mapping returned by a successful call to loadz_keymat or passed as
	an argument to loadtxt_keymat.
	'''
	# Build optional kwargs
	kwargs = { }
	if scalar is not None: kwargs['scalar'] = scalar
	if dtype is not None: kwargs['dtype'] = dtype

	try:
		mapping = loadz_keymat(f, **kwargs)
	except (ValueError, IOError):
		if nkeys is not None: kwargs['nkeys'] = nkeys
		return loadtxt_keymat(f, **kwargs)

	if nkeys is not None and len(mapping):
		key = next(mapping.keys())

		try: nk = len(key)
		except TypeError: nk = 1

		if nkeys != nk:
			raise ValueError('Cardinality of keys in mapping does not match nkeys parameter')

	return mapping


def savez_keymat(f, mapping, sortrows=True, compressed=False, comment=None):
	'''
	Stores mapping, which maps one or more integers to one or more
	numerical values, into f (which may be a string providing a file name,
	or an open file-like object) using numpy.savez (if compressed is
	False) or numpy.savez_compressed (if compressed is True).

	All keys must contain the same number of integers. Each value in the
	mapping may consiste of an arbitrary number of numeric values.

	If sortrows is True, the data will be stored in an order determined by
	sorted(mapping.keys()). Otherwise, the row order is either arbitrary or
	enforced by the input map (e.g., an OrderedDict).

	The saved npz file contains three arrays: 'keys', an N-by-M integer
	array such that each row specifies an M-integer key in the input
	mapping; 'values', which stores the values of the mapping flattened
	according to the order of 'keys', and 'lengths', which specifies the
	length of the value array for each associated key. That is,

		mapping[keys[i]] = values[start:start+lengths[i]],

	where start = sum(lengths[j] for 0 <= j < i).

	If the lengths of the value lists for all keys are the same, the
	'lengths' array may be just a scalar value, in which case 'lengths[i]'
	should be interpreted as '([lengths] * len(keys))[i]'.

	If comment is not None, it should be a string that will be stored as an
	extra array, called 'comment', in the output file. The comment will be
	ignored when loading the file.
	'''
	# Make sure any comment is a string
	if comment is not None: exargs = { 'comment': str(comment) }
	else: exargs = { }

	keys = sorted(mapping.keys()) if sortrows else list(mapping.keys())

	# Build the length array and flattened value array
	lengths, values = [ ], [ ]
	for k in keys:
		v = mapping[k]

		try:
			lengths.append(len(v))
			values.extend(v)
		except TypeError:
			lengths.append(1)
			values.append(v)

	lengths = np.array(lengths)
	values = np.array(values)

	# Collapse lengths to scalar if possible
	try: lv = lengths[0]
	except IndexError: lv = 0
	if np.all(lengths == lv):
		lengths = np.array(lv)


	# Verify the value array
	if not np.issubdtype(values.dtype, np.number):
		raise TypeError('Values in mapping must be numeric')

	# Verify the key array
	keys = np.array(keys)
	if not np.issubdtype(keys.dtype, np.integer) or keys.ndim > 2:
		raise TypeError('Keys in mapping consist of one more integers and must have consistent cardinality')

	savez = np.savez_compressed if compressed else np.savez
	savez(f, keys=keys, values=values, lengths=lengths, **exargs)


def loadz_keymat(*args, **kwargs):
	'''
	Load and return, using numpy.load(*args, **kwargs), a mapping (created
	with savez_keymat) from one or more integers to one or more numerical
	values.

	If the number of elements in every value array is 1, setting an
	optional keyword argument scalar (True by default) to False will
	preserve the values as 1-element Numpy arrays. Otherwise, 1-element
	Numpy arrays will be collapsed to scalars. The scalar keyword argument
	is stripped from the kwargs and is not passed to numpy.load.

	The data types of the value arrays can be forced by specifying an
	optional keyword argument dtype. The dtype argument will be stripped
	from the kwargs and is not passed to numpy.load.

	The returned mapping is an OrderedDict that preserves the ordering of
	keys in the input file.

	If the loaded file does not contain a valid mapping in the style
	prepared by savez_keymat, a ValueError will be raised.

	If the file contains a "comment" key, it will be silently ignored.
	'''
	# Pull specialty kwargs
	scalar = kwargs.pop('scalar', True)
	dtype = kwargs.pop('dtype', None)

	try:
		# Load the file
		with np.load(*args, **kwargs) as data:
			try:
				files = set(data.keys())

				# Ignore a comment in the file
				try: files.remove('comment')
				except KeyError: pass

				# Make sure all other fields are recognized
				if files != { 'keys', 'values', 'lengths' }: raise ValueError
			except (AttributeError, ValueError):
				raise ValueError('Unrecognized data structure in input')

			keys = data['keys']
			values = data['values']
			lengths = data['lengths']
	except AttributeError:
		raise ValueError('Invalid file format')

	# Convert the data type if desired
	if dtype is not None:
		values = values.astype(dtype)

	if not np.issubdtype(keys.dtype, np.integer) or not 0 < keys.ndim < 3:
		raise ValueError('Invalid mapping key structure')

	if not np.issubdtype(lengths.dtype, np.integer) or lengths.ndim > 1:
		raise ValueError('Invalid mapping length structure')

	if not np.issubdtype(values.dtype, np.number) or values.ndim != 1:
		raise ValueError('Invalid mapping value structure')

	if lengths.ndim == 1 and len(lengths) != len(keys):
		raise ValueError('Mapping lengths and keys do not have equal lengths')

	nvals = np.sum(lengths) if lengths.ndim == 1 else (lengths * len(keys))
	if len(values) != nvals:
		raise ValueError('Mapping values do not have appropriate lengths')

	if scalar:
		# Determine whether the mapped values can be collapsed to scalars
		if lengths.ndim == 0:
			scalar = lengths == 1
		else:
			scalar = (lengths.shape[0] > 0 and
					all(lv == 1 for lv in lengths))

	# Collapse 1-element keys to scalars
	try: keys = keys.squeeze(axis=1)
	except ValueError: pass

	if keys.ndim == 2:
		# Convert a list of key values to a tuple of Python scalars
		keys = [ tuple(k.tolist()) for k in keys ]
	else:
		# Collapse a single key value to a single Python scalar
		keys = [ k.tolist() for k in keys ]

	mapping = OrderedDict()
	start = 0

	for key, lv in zip(keys, lengths if lengths.ndim == 1 else repeat(lengths)):
		mapping[key] = values[start] if scalar else values[start:start+lv]
		start += lv

	return mapping


def loadtxt_keymat(*args, **kwargs):
	'''
	Loads a textual Numpy matrix by calling numpy.loadtxt(*args, **kwargs),
	then converts the output to an OrderedDict mapping integers in some
	positive number of leading columns to Numpy arrays composed of the
	remaining columns. The ouput dictionary preserves the ordering of rows
	in the input file.

	If the number of remaining columns is 1, setting an optional keyword
	argument scalar (default: True) to False will preserve 1-element Numpy
	arrays as the values of the dictionary. Otherwise, 1-element Numpy
	arrays in the dictionary values will be collapsed to scalars. The
	scalar keyword argument is stripped from kwargs and is not passed to
	numpy.loadtxt.

	The dimensionality of the text matrix will be forced to 2 by adding
	ndmin=2 to the kwargs. Therefore, this value should not be specified in
	args or kwargs.

	An optional keyword argument, nkeys (default: 1), will be stripped from
	kwargs to determine the number of leading columns to use as keys. If
	nkeys is 1, the keys will be single integers. For nkeys > 1, the keys
	will be tuples of integers.
	'''
	# Pull speciality kwargs
	nkeys = _strict_nonnegative_int(kwargs.pop('nkeys', 1), positive=True)
	scalar = kwargs.pop('scalar', True)

	# Ensure the dimensionality is correctly specified
	kwargs['ndmin'] = 2
	mat = np.loadtxt(*args, **kwargs)

	_, ncol = mat.shape

	if nkeys >= ncol:
		raise ValueError('Number of key columns must be less than number of columns in matrix')

	def kvmaker(g):
		k = tuple(_strict_int(gv) for gv in g[:nkeys])
		v = g[nkeys:]
		if len(k) < 2: k = k[0]
		if scalar and len(v) < 2: v = v[0]
		return k, v

	return OrderedDict(kvmaker(g) for g in mat)


def savetxt_keymat(*args, **kwargs):
	'''
	Stores a dictionary mapping integers to sequences as a textual Numpy
	matrix using numpy.savetxt(*args, **kwargs), where the keys become the
	leading columns of the matrix and the remaining columns are populated
	by the corresponding values.

	If a format is specified as the 'fmt' argument to savetxt, it must
	account for the extra columns populated by the keys.

	If kwargs contains a 'sortrows' argument, the Boolean value (defaulting
	to True) for the argument determines whether the mapping is sorted by
	keys prior to output. Without sorting, the row order is either
	arbitrary or enforced by the input map (e.g., an OrderedDict). This
	argument is not forwarded to savetxt.
	'''
	# Pull the map
	if len(args) > 1:
		x = args[1]
	else:
		x = kwargs.pop('X')

	sortrows = kwargs.pop('sortrows', True)

	def aslist(x):
		try: return list(x)
		except TypeError: return list([x])

	rows = iter(x.items()) if not sortrows else sorted(x.items())

	# Convert the dictionary to a list of lists
	mat = [ aslist(k) + aslist(v) for k, v in rows ]

	# Overwrite the input argument for the matrix
	if len(args) > 1:
		args = tuple(a if i != 1 else mat for i, a in enumerate(args))
	else:
		kwargs['X'] = mat

	np.savetxt(*args, **kwargs)


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
	from re import compile as recomp

	if ngroups < 1:
		raise ValueError('At least one number group must be specified')

	# Build the number-matching portion
	numstr = '-([0-9]+)' * ngroups
	# Enumerate the matching groups (0 is the whole matching string)
	grpidx = tuple(range(ngroups + 1))
	# Build the regexp and filter the list of files in the directory
	regexp = recomp(r'^%s%s%s$' % (prefix, numstr, suffix))
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
	from pandas import read_csv
	# Read the data and use the reducer filter if appropriate
	data = read_csv(f, skiprows=4, header=None).values
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


class TxGroupIndex(tuple):
	'''
	A class to encapsulate and type-check transmit-index pairs.
	'''
	def __new__(cls, lidx, gidx):
		'''
		Create a new TxGroupIndex with local index lidx and
		group index gidx.
		'''
		lidx = _strict_nonnegative_int(lidx)
		gidx = _strict_nonnegative_int(gidx)
		return tuple.__new__(cls, (lidx, gidx))
	@property
	def idx(self): return self[0]
	@property
	def grp(self): return self[1]

	def signForTx(self, transmission, group):
		'''
		Return the sign (-1, 0, 1) of the given transmission
		number and group for this transmit and group index.
		'''
		# If the groups don't match, the sign is zero
		if group != self.grp: return 0

		# Count number of common bits in transmission and idx
		txcom = _strict_nonnegative_int(transmission) & self.idx
		count = 0
		while txcom:
			txcom &= txcom - 1
			count += 1

		# Sign is +1 for even number of common bits
		return 1 - 2 * (count % 2)


class TxGroupConfiguration(tuple):
	'''
	A class to encapsulate and type-check transmit-group configurations.
	'''
	def __new__(cls, count, size):
		'''
		Create a new TxGroupConfiguration.
		'''
		count = _strict_nonnegative_int(count)
		size = _strict_nonnegative_int(size)
		return tuple.__new__(cls, (count, size))

	@property
	def count(self): return self[0]
	@property
	def size(self): return self[1]
	@property
	def maxtx(self): return self[0] * self[1]


class RxChannelHeader(tuple):
	'''
	A class to encapsulate and type-check receive-channel headers
	in WaveformSet files.
	'''
	def __new__(cls, idx, pos, win, txgrp=None):
		'''
		Create a new header for receive channel idx,
		element location pos = (px, py, pz), and data window
		win = (start, length). The transmit group txgrp may
		either be None or (index, group).
		'''
		from .sigtools import Window
		idx = _strict_nonnegative_int(idx)
		px, py, pz = pos
		pos = tuple(float(p) for p in (px, py, pz))
		# Force the window start to be nonnegative
		win = Window(win, nonneg=True)
		if txgrp is not None: txgrp = TxGroupIndex(*txgrp)
		return tuple.__new__(cls, (idx, pos, win, txgrp))
	@property
	def idx(self): return self[0]
	@property
	def pos(self): return self[1]
	@property
	def win(self): return self[2]
	@property
	def txgrp(self): return self[3]

	def copy(self, **kwargs):
		"Copy the header, optionally replacing certain properties."
		keys = ['idx', 'pos', 'win', 'txgrp']
		props = dict((key, kwargs.pop(key, getattr(self, key))) for key in keys)
		if len(kwargs):
			raise TypeError("Unrecognized keyword '%s'" % (next(kwargs.keys()),))
		return type(self)(**props)


class WaveformSet(object):
	'''
	A class to encapsulate a (possibly multi-facet) set of pulse-echo
	measurements from a single target.
	'''
	# A bidirectional mapping between typecodes and Numpy dtype names
	from pycwp.util import bidict
	typecodes = bidict(I2 = 'int16', I4 = 'int32', I8 = 'int64',
			F2 = 'float16', F4 = 'float32', F8 = 'float64',
			C4 = 'complex64', C8 = 'complex128')

	@classmethod
	def fromfile(cls, f, *args, **kwargs):
		'''
		Create a new WaveformSet object and use load() to populate the
		object with the contents of the specified file (a file-like
		object or a string naming the file).

		Extra args and kwargs are passed to the load() method.
		'''
		# Create an empty set, then populate from the file
		wset = cls()
		wset.load(f, *args, **kwargs)
		return wset


	@classmethod
	def fromwaveform(cls, wave, copy=False, hdr=None, tid=0, f2c=0):
		'''
		Create a new WaveformSet object with a single transmit index
		and a single receive index with a sample count and data type
		defined by the provided Waveform wave. The sole waveform record
		will be populated with wave.

		If copy is False, the record in the WaveformSet will, whenever
		possible, capture a reference to the waveform data instead of
		making a copy. If copy is True, a copy will always be made.

		If hdr is not None, it should be a receive-channel header that
		will be used for the single receive-channel record in the
		output WaveformSet. The value of hdr.win will be overwritten
		with wave.datawin. If hdr is None, a default value

			(0, [0., 0., 0.], wave.datawin)

		will be used.

		The parameter tid should be a single nonnegative integer that
		specifies the transmit index to assign to the Waveform.

		The parameter f2c should be a single nonnegative integer that
		specifies the fire-to-capture delay to encode in the set.
		'''
		# Create the set
		wset = cls(1, tid, wave.nsamp, f2c, wave.dtype)

		if hdr is None:
			# Create a default header
			hdr = RxChannelHeader(0, [0.]*3, wave.datawin)
		else:
			# Ensure hdr is RxChannelHeader, then set datawin
			hdr = RxChannelHeader(*hdr).copy(win=wave.datawin)

		wset.setrecord(hdr, wave.getsignal(wave.datawin), copy)
		return wset


	@classmethod
	def concatenate(cls, *args, **kwargs):
		'''
		Create a new WaveformSet object that concatenates all
		transmit-receive waveforms from the input WaveformSets provided
		in *args. The concatenated txidx list is always sorted
		regardless of input ordering.

		By default, the input waveforms must collectively contain
		exactly one waveform for each transmit-receive product in the
		output. Setting the sole optional kwarg defzero=True overrides
		this default to allow missing transmit-receive waveforms to be
		treated as zero.

		Duplicate transmit-receive waveforms among the inputs is always
		an error.
		'''
		from habis.sigtools import Window
		if len(args) < 2:
			raise TypeError('At least two input WaveformSets must be specified')

		defzero = kwargs.pop('defzero', False)

		if len(kwargs):
			raise TypeError('Unrecognized keyword %s' % (next(kwargs.keys()),))

		# Map the nsamp and f2c for each file to a global range
		lsamp = 0
		f2c = float('inf')

		# Map receive channels to transmit-channel lists and receive-channel headers
		rxtxmap = defaultdict(set)
		rcvhdrs = { }

		# Ensure dtype and transmit-group configurations are compatible
		dtype = args[0].dtype
		txgrps = args[0].txgrps

		for wset in args:
			f2c = min(f2c, wset.f2c)
			lsamp = max(lsamp, wset.nsamp + wset.f2c)

			ltx = set(wset.txidx)

			if wset.dtype != dtype:
				raise TypeError('All WaveformSets must have the same datatype')

			if wset.txgrps != txgrps:
				raise TypeError('All WaveformSets must have the same Tx-group configuration')

			for hdr in wset.allheaders():
				rxi = hdr.idx

				if not rxtxmap[rxi].isdisjoint(ltx):
					raise ValueError('Receive channel %d contains duplicate waveforms for at least one transmission' % rxi)

				# Map this record to a data window with 0 f2c
				hwin = Window(hdr.win.start + wset.f2c, hdr.win.length)

				try:
					rhdr = rcvhdrs[rxi]
				except KeyError:
					# Make the data window relative to 0 f2c
					rcvhdrs[rxi] = hdr.copy(win=hwin)
				else:
					if not np.allclose(rhdr.pos, hdr.pos):
						raise ValueError('Inconsistent positions for receive channel %d' % rxi)
					if rhdr.txgrp != hdr.txgrp:
						raise ValueError('Inconsistent transmit-group configuration for receive channel %d' % rxi)

					# Find encompassing window for this channel
					rwin = rhdr.win
					cwin = Window(min(rwin.start, hwin.start),
							end=max(rwin.end, hwin.end))
					rcvhdrs[rxi] = rhdr.copy(win=cwin)

				rxtxmap[rxi].update(ltx)

		# Identify all unique transmit channels
		txidx = { v for s in rxtxmap.values() for v in s }

		if not defzero:
			# Check to make sure every waveform is accounted for
			if not all(v == txidx for v in rxtxmap.values()):
				raise ValueError('Not all receive channels specify waveforms for all transmissions')

		# Build the transmit-index list and map to record rows
		txidx = sorted(txidx)
		txmap = { v: i for i, v in enumerate(txidx) }
		ntx = len(txidx)

		# Create and populate the output WaveformSet
		outset = cls(ntx, 0, lsamp - f2c, f2c, dtype, txgrps)
		outset.txidx = txidx

		for rxi, rhdr in sorted(rcvhdrs.items()):
			# Subtract the global f2c from the data window
			win = Window(rhdr.win.start - outset.f2c, rhdr.win.length)
			rec = np.zeros((ntx, win.length), dtype=outset.dtype)

			for wset in args:
				try: lhdr, dat = wset.getrecord(rxi)
				except KeyError: continue

				# Map local rows to global record rows
				recrows = [txmap[txi] for txi in wset.txidx]

				# Map the local data window to the global data window
				lwin = Window(lhdr.win.start + wset.f2c - rhdr.win.start, lhdr.win.length, nonneg=True)

				# Copy the data
				rec[recrows,lwin.start:lwin.end] = dat

			# Correct the data window in the record header
			outset.setrecord(rhdr.copy(win=win), rec, copy=False)

		return outset


	@classmethod
	def empty_like(cls, wset, with_context=True):
		'''
		Create a new instance of WaveformSet configured exactly as
		wset, except without any waveform records.

		If with_context is True, the dictionary wset.context will be
		copied (shallowly) into the created WaveformSet. Otherwise, the
		context of the created WaveformSet will be empty
		'''
		nwset = cls(wset.ntx, wset.txstart, wset.nsamp, wset.f2c, wset.dtype, wset.txgrps)
		if with_context: nwset.context = wset.context.copy()
		else: nwset.context = { }
		return nwset


	def __init__(self, ntx=0, txstart=0, nsamp=4096, f2c=0,
			dtype=np.dtype('int16'), txgrps=None):
		'''
		Create an empty WaveformSet object that embodies acquisitions
		of a set of waveforms from a total of ntx transmission indices (0-based)
		starting from index txstart. Each acquisition starts after a
		fire-to-capture delay of f2c samples and persists for nsamp
		samples. Waveform arrays are stored with the specified Numpy
		dtype.

		If txgrps is specified, it should be a TxGroupConfiguration
		object or a tuple of the form (count, size) that specifies the
		number of transmit groups into which transmissions are
		subdivided, and the number of elements in each group.
		'''
		# Record the waveform dtype
		self._dtype = np.dtype(dtype)

		# Prepopulate properties that will be validated later
		self._f2c = 0
		self._nsamp = 0
		self._ntx = 0
		self._txstart = 0
		self._txgrps = None

		# Create an empty, ordered record dictionary
		# Needed for validation of other properties
		self._records = OrderedDict()

		# Create an empty group map
		self._groupmap = { }

		# Assign validated properties
		self.nsamp = nsamp
		self.f2c = f2c

		# Build and validate the transmit-channel mapping
		self.ntx = ntx
		self.txstart = txstart

		# Initialize the group configuration as specified
		self.txgrps = txgrps

		# Extra scan context can be read from a file header and is
		# passed on when writing compatible versions, but is never
		# inherently interpreted
		self.context = { }


	def _verify_file_version(self, version, write=False):
		'''
		Ensure that the provided version matches one supported by the
		WaveformSet class. If version is unsupported, a ValueError is
		raised. Otherwise, just return the version tuple.
		'''
		try:
			major, minor = version
			major = _strict_nonnegative_int(major)
			minor = _strict_nonnegative_int(minor)
		except (TypeError, ValueError):
			raise ValueError('Version format is not recognized')

		if major != 1: raise ValueError('Unsupported major version')

		if not write:
			# Support all currently defined formats for reading
			if not (0 <= minor < 6):
				raise ValueError('Unsupported minor version for reading')
			return (major, minor)

		# Only version-5 writes are supported
		if minor != 5:
			raise ValueError('Unsupported minor version for writing')

		return major, minor


	def store(self, f, append=False, ver=(1,5)):
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
		if isinstance(f, str):
			f = open(f, mode=('wb' if not append else 'ab'))

		# Only version (1,5) is supported for output
		major, minor = self._verify_file_version(ver, write=True)

		# A missing transmit-group configuration takes the special value (0,0)
		try: gcount, gsize = self.txgrps
		except (TypeError, ValueError): gcount, gsize = 0, 0

		if not append:
			# Encode the magic number and file version
			hbytes = struct.pack('<4s2I', 'WAVE', major, minor)

			# Encode temperature values
			temps = self.context.get('temps', [float('nan')]*2)
			hbytes += np.asarray(temps, dtype=np.float32).tobytes()

			# Encode the datatype
			typecode = self.typecodes.inverse[np.dtype(self.dtype).name][0]
			hbytes += struct.pack('<2s', typecode)

			# Encode transmission parameters
			hbytes += struct.pack('<4I2HI', self.f2c, self.nsamp,
					self.nrx, self.ntx, gcount, gsize, self.txstart)

			# Unspecified TGC parameters default to 0
			try: tgc = self.context['tgc']
			except KeyError: tgc = np.zeros(256, dtype=np.float32)

			if len(tgc) != 256:
				raise ValueError('File version (1,4) requires 256 TGC floats')

			hbytes += tgc.tobytes()

			f.write(hbytes)

		f.seek(0, 2)

		# Write each record in turn
		for idx in sorted(self.rxidx):
			hdr, waveforms = self._get_record_raw(idx)

			if idx != hdr.idx:
				raise ValueError('Record index does not match receive-channel index')

			px, py, pz = hdr.pos
			ws, wl = hdr.win

			# Without a transmit-group configuration, use (0,0)
			try: li, gi = hdr.txgrp
			except (TypeError, ValueError): li, gi = 0, 0

			# Enclode the receive-channel header
			hbytes = struct.pack('<3I3f2I', idx, li, gi, px, py, pz, ws, wl)

			f.write(hbytes)
			# Encode the waveform data
			wbytes = waveforms.tobytes()
			f.write(wbytes)
			f.flush()

		f.close()


	def load(self, f):
		'''
		Associate the WaveformSet object with the data in f, a file-
		like object or string specifying a file name. If f is a file-
		like object, parsing starts from the current file position.

		Existing waveform records will be eliminated. All parameters of
		the WaveformSet (arguments to the constructor) will be reset to
		values specified in the file header. Failure to parse the file
		completely may result in the instance being corrupted.

		Each block of waveform data is memory-mapped from the source
		file. This mapping is copy-on-write; changes do not persist.
		'''
		# Open the file if it is not open
		if isinstance(f, str):
			f = open(f, mode='rb')

		def funpack(fmt):
			sz = struct.calcsize(fmt)
			return struct.unpack(fmt, f.read(sz))

		# Read the magic number and file version
		magic, major, minor = funpack('<4s2I')

		if magic != 'WAVE':
			raise ValueError('Invalid magic number in file')

		major, minor = self._verify_file_version((major, minor))

		if minor > 4:
			# Read temperature context
			try: self.context['temps'] = np.fromfile(f, dtype=np.float32, count=2)
			except ValueError:
				raise ValueError('Temperature section of header must contain 2 floats')

		# Read the type code for this file
		typecode = funpack('<2s')[0]
		dtype = np.dtype(self.typecodes[typecode])

		# Clear the record array
		self.clearall()

		# Parse common transmission parameters
		f2c, nsamp, nrx, ntx = funpack('<4I')

		# Set the file-level parameters
		self.dtype = dtype
		self.nsamp = nsamp
		self.f2c = f2c
		self.ntx = ntx
		# By default, start the transmission indexing at 0
		self.txstart = 0

		# Clear any group configuration for now
		self.txgrps = None

		if minor > 1:
			# Read the group configuration
			count, size = funpack('<2H')
			# Make sure both values are sensible integers
			count = _strict_nonnegative_int(count)
			size = _strict_nonnegative_int(size)

			# Only configure transmit groups if the count is positive
			if count > 0:
				# Default group size, if unspecified, is 10240 / count
				if size == 0:
					size = 10240 // count
					if size * count != 10240:
						raise ValueError('Cannot infer group size for %d groups' % count)

				self.txgrps = count, size

			# For version (1,4) and above, read an explicit txstart
			if minor >= 4: self.txstart = funpack('<I')[0]

			# Read 256 TGC parameters in the header
			try: self.context['tgc'] = np.fromfile(f, dtype=np.float32, count=256)
			except ValueError:
				raise ValueError('TGC section of header must contain 256 floats')
		elif minor == 0:
			# Verion 0 uses an explicit 1-based transmit-index list
			self.txidx = np.fromfile(f, dtype=np.uint32, count=ntx) - 1

		# Record the start of the waveform data records in the file
		fsrec = f.tell()

		# Use a single Python mmap buffer for backing data
		# This avoids multiple openings of the file for independent maps
		buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
		f.seek(fsrec)

		# For (1, 2) files, keep a running index tally
		idx = -1

		# If the set isn't configured for transmit groups,
		# ignore any group spec in the receive-channel headers
		usegrps = (self.txgrps is not None)

		# Parse through the specified number of receive records
		while True:
			if minor == 2:
				# Update running index
				idx += 1
			else:
				# Read a global channel index
				# Correct 1-based indexing in early versions
				try: idx = funpack('<I')[0] - int(minor < 2)
				except struct.error: break

			# Read element position and data window parameters
			if minor > 1:
				# Also read transmission group configuration
				try: i, g, px, py, pz, ws, wl = funpack('<2I3f2I')
				except struct.error: break

				txgrp = (i, g) if usegrps else None
				if minor == 2:
					# Correct an off-by-one window specification bug
					if wl == nsamp and ws == 1: ws = 0
			else:
				try: px, py, pz, ws, wl = funpack('<3f2I')
				except struct.error: break
				txgrp = None

			# Build the channel header
			hdr = (idx, (px, py, pz), (ws, wl), txgrp)

			# Determine the shape of the waveform
			waveshape = (ntx, wl)
			# Determine the start and end of waveform data block
			fsmap = f.tell()
			try:
				# Return a view of the map
				wavemap = np.ndarray(waveshape, dtype=dtype,
						buffer=buf, order='C', offset=fsmap)
			except TypeError: break
			# Add the record to the set
			self.setrecord(hdr, wavemap, copy=False)
			# Skip to the next header
			f.seek(fsmap + wavemap.nbytes)
			# Update the offset of the next record
			fsrec = f.tell()

		import warnings

		if f.tell() != fsrec:
			warnings.warn('Junk at end of file')

		if nrx and self.nrx != nrx:
			raise ValueError('Header specifies %d records, but read %d' % (nrx, self.nrx))


	@property
	def rxidx(self):
		'''
		Return a list of receive-channel indices in file order.
		'''
		return list(self._records.keys())


	@property
	def txgrps(self):
		'''
		Return the (count, size) of transmit groups, or None for no grouping.
		'''
		return self._txgrps


	@txgrps.setter
	def txgrps(self, grps):
		'''
		Set the group count and length. Removes any existing groupmap
		property.
		'''
		if grps == self._txgrps: return

		if self.nrx > 0:
			raise ValueError('Cannot change transmit-group configuration with existing records')

		if grps is None:
			self._txgrps = None
			self.groupmap = None
			return

		try:
			grps = TxGroupConfiguration(*grps)
		except (TypeError, ValueError):
			raise ValueError('Parameter must be None or (count, size) tuple')

		if grps.maxtx < self.ntx:
			raise ValueError('Implied maximum transmission count is less than number of recorded transmissions')
		if grps.maxtx <= self.txstart:
			raise ValueError('Implied maximum transmission count is less than starting transmission index')

		self._txgrps = grps
		self.groupmap = None


	@property
	def txstart(self):
		'''
		Return the first transmission index in the records.
		'''
		return self._txstart


	@txstart.setter
	def txstart(self, txstart):
		'''
		Set the first transmission index in the records, which must be
		a nonnegative integer within the tranmission range implied by
		the group configuration in self.txgrps.
		'''
		if txstart == self._txstart: return

		txstart = _strict_nonnegative_int(txstart)

		try:
			maxtx = self.txgrps.maxtx
		except AttributeError:
			pass
		else:
			if txstart >= maxtx:
				raise ValueError('Parameter txstart exceeds maxtx of transmit-group configuration')

		self._txstart = txstart


	@property
	def txidx(self):
		'''
		Return a generator of tranmit-channel indices in file order.
		'''
		txstart = self.txstart
		txgrps = self.txgrps

		try:
			maxtx = self.txgrps.maxtx
		except AttributeError:
			for i in range(txstart, txstart + self.ntx):
				yield i
		else:
			for i in range(txstart, txstart + self.ntx):
				yield i % maxtx


	@txidx.setter
	def txidx(self, txidx):
		'''
		Checks the provided list for sequential ordering of the input
		sequence txidx and, if the check is satisfied, assigns
		self.txstart and self.ntx accordingly.

		If the indices are not sequential, but self.txgrps is None, the
		txgrp configuration and self.groupmap will be set to map
		transmit indices 0 through len(txidx) - 1 to the elements of
		txidx.
		'''
		txidx = list(txidx)

		try: txstart = txidx[0]
		except IndexError:
			self.ntx = 0
			self.txstart = 0
			return

		try:
			maxtx = self.txgrps.maxtx
		except AttributeError:
			def nextval(x): return (x + 1)
		else:
			def nextval(x): return (x + 1) % maxtx

		last = txstart
		sequential = True

		for nv in txidx[1:]:
			last = nextval(last)
			if nv != last:
				sequential = False
				break

		def atomic_set(txstart, ntx):
			# Record the old txstart to ensure atomicity
			otxstart = self.txstart
			self.txstart = txstart

			try: self.ntx = ntx
			except:
				# Restore the old txstart before failing
				self.txstart = otxstart
				raise

		if not sequential:
			if self.txgrps is not None:
				raise ValueError('Indices must be sequential or wrap when txgrps is defines')
			# Set txgrp configuration to remap out-of-sequence indices
			atomic_set(0, len(txidx))
			self.txgrps = (self.ntx, 1)
			self.groupmap = { txi: (0, i) for i, txi in enumerate(txidx) }
		else:
			atomic_set(txstart, len(txidx))


	@property
	def ntx(self):
		'''
		Return the number of transmissions per receive channel.
		'''
		return self._ntx


	@ntx.setter
	def ntx(self, ntx):
		'''
		Set the number of transmissions per receive channel.
		'''
		# Take no action if the count hasn't changed
		if ntx == self._ntx: return

		# Don't attempt to change the transmit count with existing records
		if self.nrx > 0:
			raise ValueError('Cannot change number of transmissions with existing records')

		try:
			if ntx > self.txgrps.maxtx:
				raise ValueError('Number of transmissions must not exceed maxtx implied by transmit-group configuration')
		except AttributeError:
			pass

		self._ntx = _strict_nonnegative_int(ntx)


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
		'''
		Set the datatype used to store waveforms.
		'''
		if self._dtype == value: return

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
	def nsamp(self, nsamp):
		'''
		Set the total number of samples in the acquisition window.
		Ensure existing records don't fall outside of the window.
		'''
		if self._nsamp == nsamp: return

		# Force the new value to be an nonnegative integer
		nsamp = _strict_nonnegative_int(nsamp)

		# Check all existing records to ensure their windows don't
		# extend past the new acquisition window
		for hdr, wforms in self.allrecords():
			start, length = hdr.win
			if start + length > nsamp:
				raise ValueError('Acquisition window fails to contain stored waveforms')

		# Set the new value
		self._nsamp = nsamp


	@property
	def f2c(self):
		'''
		Return the fire-to-capture delay in 20-MHz samples.
		'''
		return self._f2c


	@f2c.setter
	def f2c(self, val):
		'''
		Set the fire-to-capture delay in 20-MHz samples.
		'''
		if self._f2c == val: return
		self._f2c = _strict_nonnegative_int(val)


	@property
	def groupmap(self):
		'''
		Access a copy of the map between global element indices to
		tuples (local index, group index) that govern firing order.
		'''
		return dict(self._groupmap)


	@groupmap.setter
	def groupmap(self, grpmap):
		'''
		Check the provided mapping from global element indices to
		(local index, group index) for consistency and assign the map
		to this instance.

		Set grpmap to None or an object with 0 len() to clear the map.
		'''
		if grpmap is None or len(grpmap) < 1:
			self._groupmap = { }
			return

		if self.txgrps is None:
			raise ValueError('Cannot set a group map without a txgrps configuration for the WaveformSet')

		# Make sure the map is valid and consistent with txgrp configuration
		ngrpmap = { }
		for k, v in grpmap.items():
			ki = _strict_nonnegative_int(k)
			vi, vg = [_strict_nonnegative_int(vl) for vl in v]
			if vi >= self.txgrps.size:
				raise ValueError('Local index in group map exceeds txgrp size')
			if vg >= self.txgrps.count:
				raise ValueError('Group index in group map exceeds txgrp count')
			ngrpmap[ki] = (vi, vg)

		# Check any local receive-channels for consistence
		for hdr in self.allheaders():
			if ngrpmap.get(hdr.idx, hdr.txgrp) != hdr.txgrp:
				raise ValueError('Group map does not match receive-channel record at index %d' % hdr.idx)

		self._groupmap = ngrpmap


	def element2tx(self, elt, unfold=True):
		'''
		Convert an element index elt into a transmission index. If no
		transmit-group configuration exists, this is *ALWAYS* the
		identity map.

		When a transmit-group configuration exists, self.groupmap is
		first checked for a transmit index for elt. If the groupmap
		does not exist or fails to specify the necessary index, the
		txgrp configuration for a receive-channel record for index elt
		(if one exists) is used.

		If unfold is True, the transmission index is a scalar value
		that directly indexes rows in record arrays. If unfold is
		False, the transmission index is a pair (locidx, grpnum) that
		maps to the unfolded index, t, by

			t = locidx + grpnum * self.txgrps.gsize.
		'''
		elt = _strict_nonnegative_int(elt)

		try: gcount, gsize = self.txgrps
		except TypeError: return elt

		try:
			txgrp = self._groupmap[elt]
		except KeyError:
			try: txgrp = self.getheader(elt).txgrp
			except KeyError:
				raise KeyError('Could not find map record for receive channel %d' % elt)

		try:
			idx, grp = txgrp
		except (TypeError, ValueError) as e:
			raise ValueError('Unable to unpack invalid txgrp for channel %d' % elt)

		return (grp * gsize + idx) if unfold else (idx, grp)


	def tx2row(self, tid):
		'''
		Convert a transmit-channel index into a waveform-array row index.
		'''
		# Ensure that the argument is properly bounded
		tid = _strict_nonnegative_int(tid)

		txstart = self.txstart

		try: maxtx = self.txgrps.maxtx
		except AttributeError: maxtx = None

		if maxtx is not None:
			if tid >= maxtx:
				raise ValueError('Argument tid exceeds self.txgrps.maxtx')
			# Shift low values to account for wraparound
			if tid < txstart: tid += maxtx

		# Shift relative to start
		tid -= self.txstart

		# Ensure the bounds are sensible
		if not 0 <= tid < self.ntx:
			raise ValueError('Transmit index is not contained in this file')
		return tid


	def _get_record_raw(self, rid):
		'''
		Return the raw (header, data) record for a given receive
		channel rid, with only sanity checks on rid.
		'''
		return self._records[_strict_nonnegative_int(rid)]


	def getheader(self, rid):
		'''
		Return the channel header for receive channel rid.
		'''
		return self._get_record_raw(rid)[0]


	def getrecord(self, rid, tid=None, window=None, dtype=None, maptids=False):
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

		If maptids is True, any indices specified in tid will be
		converted from an element index to a transmission index using
		self.element2tx().
		'''
		# Grab receive record, copy header to avoid corruption
		hdr, waveforms = self._get_record_raw(rid)

		if maptids and tid is not None:
			# Map the transmit indices to element indices
			try:
				tid = self.element2tx(tid)
			except TypeError:
				tid = [self.element2tx(t) for t in tid]

		try:
			tcidx = self.tx2row(tid)
			singletx = True
		except TypeError:
			singletx = False
			if tid is None:
				tcidx = list(range(self.ntx))
			else:
				tcidx = [self.tx2row(t) for t in tid]

		if window is None:
			if dtype is None:
				# With no type override, just return a view
				return hdr, waveforms[tcidx,:]
			else:
				# Force a type conversion and copy
				if dtype == 0:
					dtype = waveforms.dtype
				return hdr, waveforms[tcidx,:].astype(dtype, copy=True)

		# Handle a specific data window
		from .sigtools import Window
		window = Window(window)

		# Handle unspecified data types
		if dtype is None or dtype == 0:
			dtype = waveforms.dtype

		# Create an output array to store the results
		oshape = (1 if singletx else len(tcidx), window.length)
		output = np.zeros(oshape, dtype=dtype)

		try:
			# Figure out the overlapping sample window
			# Raises TypeError if overlap() returns None
			from pycwp.cutil import overlap
			ostart, istart, wlen = overlap(window, hdr.win)
			oend, iend = ostart + wlen, istart + wlen

			# Copy portion of waveforms overlapping the window
			output[:,ostart:oend] = waveforms[tcidx,istart:iend]
		except TypeError: pass

		# For a scalar tid, collapse the 2-D array
		if singletx: output = output[0]

		# Override the window in the header copy
		return hdr.copy(win=window), output


	def getwaveform(self, rid, tid, *args, **kwargs):
		'''
		Return, as one or more habis.sigtools.Waveform objects, the
		waveform(s) recorded at receive-channel index rid from the
		(scalar or iterable of) transmission(s) tid.

		If tid is a scalar, a single Waveform object is returned.
		Otherwise, is tid is an iterable or None (which pulls all
		transmissions), a list of Waveform objects is returned.

		If a keyword-only argument '' is

		Extra args and kwargs are passed through to getrecord().
		'''
		from .sigtools import Waveform
		# Grab the relevant row of the record
		hdr, wform = self.getrecord(rid, tid, *args, **kwargs)

		# Wrap a single desired signal in a Waveform object
		if np.ndim(wform) == 1:
			return Waveform(self.nsamp, wform, hdr.win.start)
		else:
			return [Waveform(self.nsamp, w, hdr.win.start) for w in wform]


	def __getitem__(self, key):
		'''
		For a scalar key, return self.record(key).

		For a key (rid, tid), return self.getwaveform(rid, tid).

		All other inputs are invalid.
		'''
		# Handle a single-integer index
		try: len(key)
		except TypeError: return self.getrecord(rid)

		# Split two-integer indices
		try: rid, tid = key
		except ValueError:
			raise TypeError('Item key should be exactly one or two integers')

		return self.getwaveform(rid, tid)


	def delrecord(self, rid):
		'''
		Delete the waveform record for the receive-channel index rid.
		'''
		del self._records[_strict_nonnegative_int(rid)]


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

		If the header specifies None for txgrp, but the WaveformSet
		transmit-group configuration is not None, any groupmap
		associated with the WaveformSet will be searched for a matching
		receive-channel index to create a matching txgrp. No other
		automatic txgrp manipulation is attempted.

		The waveform array must either be a Numpy ndarray or None. When
		waveforms takes the special value None, a new, all-zero
		waveform array is created (regardless of the value of copy).

		If copy is False, a the record will store a reference to the
		waveform array if the types are compatible. If copy is True, a
		local copy of the waveform array, cast to this set's dtype,
		will always be made.
		'''
		hdr = RxChannelHeader(*hdr)

		if self.txgrps is not None:
			# Ensure consistency with the group configuration
			if hdr.txgrp is None:
				# Check the group map for a matching record
				try:
					txgrp = self.element2tx(hdr.idx, unfold=False)
				except (KeyError, TypeError):
					raise ValueError('Record is missing required txgrp configuration')
				else:
					hdr = hdr.copy(txgrp=txgrp)
			elif hdr.txgrp.grp >= self.txgrps.count:
				raise ValueError('Record group number too large')
			elif hdr.txgrp.idx >= self.txgrps.size:
				raise ValueError('Record local index too large')
			else:
				# Ensure consistency with the groupmap
				try:
					rgrp = self.groupmap[hdr.idx]
				except (TypeError, KeyError):
					pass
				else:
					if rgrp != hdr.txgrp:
						raise ValueError('Record txgrp does not match groupmap')
		elif hdr.txgrp is not None:
			raise ValueError('Record contains inappropriate txgrp configuration')

		# Check that the header bounds make sense
		if hdr.win.end > self.nsamp:
			raise ValueError('Waveform sample window exceeds acquisition window duration')

		if waveforms is None:
			# Create an all-zero waveform array
			wshape = (self.ntx, hdr.win.length)
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
			if nsamp != hdr.win.length:
				raise ValueError('Waveform array does not match sample count specified in header')

		# Add or replace the record
		self._records[hdr.idx] = (hdr, waveforms)


	def setwaveform(self, rid, tid, wave, maptid=False):
		'''
		Replace the waveform at receive index rid and transmit index
		tid with the provided habis.sigtools.Waveform wave. When replacing
		the existing waveform, the signal will be padded and clipped as
		necessary to fit into the window defined for the record.

		If maptid is True, the tid will be converted from an element
		index to a transmission index using self.element2tx().
		'''
		if maptid: tid = self.element2tx(tid)

		tcidx = self.tx2row(tid)

		# Pull the existing record
		hdr, wfrec = self._get_record_raw(rid)

		# Overwrite the transmit row with the input waveform
		wfrec[tcidx,:] = wave.getsignal(window=hdr.win, dtype=wfrec.dtype)


	def allrecords(self, *args, **kwargs):
		'''
		Return a generator that fetches each record, in channel-index
		order, using self.getrecord(rid, window, dtype).
		'''
		for rid in sorted(self.rxidx):
			yield self.getrecord(rid, *args, **kwargs)


	def allheaders(self):
		'''
		Return a generator that fetches, in channel-index order, only
		the receive-channel record headers.
		'''
		for rid in sorted(self.rxidx):
			yield self.getheader(rid)
