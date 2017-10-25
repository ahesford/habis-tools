#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, os
from subprocess import call

from pycwp import process

def procexec(prog, args, inputs):
	'''
	If the sequence inputs is not empty, execute the script

		<prog> *args, *inputs

	If inputs is empty, take no action. If args is empty, ignore the arguments.
	'''
	# There is no work to undertake if there are no inputs
	if len(inputs) < 1: return
	# Build the argument list, starting with the program
	pargs = [prog] + [str(a) for a in args]
	# Call the program wth the inputs converted to strings and appended)
	call(pargs + [str(a) for a in inputs])


def facetindex(fname):
	'''
	Pull the facet index from the provided filename, which must have an
	extension of the form

		.facet<index>.<suffix>

	for an arbitrary suffix and a numerical index in the facet section.
	'''
	# Grab the facet label
	facetlabel = fname.split('.')[-2]
	# Pull the integer part from the text
	return int(facetlabel.replace('facet', ''), base=10)


if __name__ == '__main__':
	# The default name for the EvaluateScattering CUDA program
	defprogname = 'EvaluateScattering'
	# The environment variable checked for the CUDA program path
	evalscatenv = 'EVALUATESCATTERING'
	
	if len(sys.argv) < 6:
		print('USAGE: %s <gpus> <freq> <elemlocs> <sensitivity> <input> ...' % sys.argv[0], file=sys.stderr)
		print('  File names MUST end with an extension .facet<index>.<suffix> for an', file=sys.stderr)
		print('  arbitrary extension <ext> (no dots) and an integer <idx> between 0 and 39', file=sys.stderr)
		print('', file=sys.stderr)
		print('  Override the environment variable %s with the path to' % evalscatenv, file=sys.stderr)
		print('  the %s CUDA program (default: $HOME/bin/EvaluateScattering)' % defprogname, file=sys.stderr)
	
		sys.exit(128)
	
	try:
		# Try to grab the CUDA program from the environment
		evalscat = os.environ[evalscatenv]
	except KeyError:
		try:
			# Otherwise, try $HOME/bin/EvaluateScattering
			bindir = os.path.join(os.environ['HOME'], 'bin')
		except KeyError:
			# If $HOME is undefined, use the local directory
			print('WARNING: $HOME undefined, trying local directory for %s' % defprogname, file=sys.stderr)
			bindir = '.'
	
		evalscat = os.path.join(bindir, defprogname)
	
	print('Trying %s for %s CUDA program' % (evalscat, defprogname), file=sys.stderr)
	
	# Grab the list of GPUs to use
	gpus = [int(s.strip()) for s in sys.argv[1].split(',')]
	ngpus = len(gpus)
	
	# Grab the frequency and the names of the element location and sensitivity files
	freq, elemlocs, elemsens = sys.argv[2:5]
	# Grab the remaining files and pull out each file's associated facet index
	scatfiles = [(f, facetindex(f)) for f in sys.argv[5:]]
	
	# The only positional argument for subprocesses is the executable name
	args = [evalscat]
	
	# Process files in groups corresponding to common facet indices
	for fidx in set(f[-1] for f in scatfiles):
		# Grab all files that share the current facet
		facetfiles = [f[0] for f in scatfiles if f[1] == fidx]
		with process.ProcessPool() as pool:
			for i, g in enumerate(gpus):
				kwargs = {}
				# In the "args" argument, pass the device ID, frequency, source
				# facet index, and the element location and sensitivity files
				kwargs['args'] = [g, freq, fidx, elemlocs, elemsens]
				# Inputs the "inputs" argument, pass a share of the inputs
				kwargs['inputs'] = facetfiles[i::ngpus]
				pool.addtask(target=procexec, args=args, kwargs=kwargs)
			pool.start()
			pool.wait()
