'''
Routines for manipulating HABIS data file formats.
'''

import numpy as np
import re, os
import pandas

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


def readballfile(f, retwin=None):
	'''
	Given a file f, which may be an already-open file object or a string
	specifying the name, read the waveform data measured at 512 receive
	channels from transmissions from each of the 512 transmit channels.

	If retwin is not None, it should be a tuple (start, length) that
	specifies the start and length of the temporal window over which all
	waveforms will be read.

	Each receive channel has a header that specifies, in this order:

		1. Transmit index (int, recorded as float32)
		2. Element position in the hemispheric array (3 float32)
		   - Coordinate origin is at center of hemisphere
		   - Coordinates may have an arbitrary (x, y) offset
		   - The z axis points "out of the bowl" of the hemisphere
		3. Index of first sample of recorded window (int32)
		   - Sample 0 corresponds to time immediately after transmit
		   - Sampling frequency is 20 MHz
		4. Number of samples, nt, in each captured waveform (int32)

	Following the header, 512 received waveforms (one per transmission) are
	stored as a block of (512 * nt) int16 values indicating ADC counts.
	Receive channel data blocks (header + waveforms) are concatenated in
	the file.

	The return value is a tuple containing, in order:

		1. List of 512 transmit indices (as ints)
		2. List of 512 element-coordinate 3-tuples (x, y, z)
		3. 2-tuple (start, length) of expanded time window
		   - Time window is smallest possible window that fully
		     contains the recording windows for every receive channel
		4. 512 x 512 x Ns array of waveforms
		   - Transmit axis varies along first axis
		   - Receive axis varies along second axis
		   - Ns is length of expanded time window
		   - Values read as int16 are converted to float32
	'''
	# Open the input file if it isn't already open
	if isinstance(f, (str, unicode)): f = open(f, mode='rb')

	# Record the transmit indices, element coordinates and time windows
	twins = []
	txidx = []
	elcrd = []
	# Also store each waveform block
	waveforms = []

	for i in range(512):
		# The Tx index is 1-based in the file; adjust
		idx = np.fromfile(f, count=1, dtype=np.float32)[0]
		txidx.append(int(idx) - 1)
		# Convert coordinates to Python floats
		crd = np.fromfile(f, count=3, dtype=np.float32)
		elcrd.append(tuple(float(c) for c in crd))
		# Grab the local window, convert to Python ints
		win = tuple(np.fromfile(f, count=2, dtype=np.int32))
		twins.append(tuple(int(w) for w in win))
		# Read the waveforms for this receive channel
		wform = np.fromfile(f, count=512 * win[-1], dtype=np.int16)
		waveforms.append(wform.reshape((512, win[-1]), order='C'))

	# If a return window was not specified, figure out the minimum necessary
	if retwin is None:
		start = min(w[0] for w in twins)
		end = max(w[0] + w[1] for w in twins)
		retwin = (start, end - start)

	# Create an array to store the concatenated waveforms
	outwaves = np.zeros((512, 512, retwin[-1]), dtype=np.float32)

	for ridx, (win, waves) in enumerate(zip(twins, waveforms)):
		istart = max(0, retwin[0] - win[0])
		ostart = max(0, win[0] - retwin[0])
		iend = min(win[1], retwin[0] + retwin[1] - win[0])
		oend = min(retwin[1], win[0] + win[1] - retwin[0])
		# Record the waveforms in the output array
		outwaves[:,ridx,ostart:oend] = waves[:,istart:iend]

	return txidx, elcrd, retwin, outwaves
