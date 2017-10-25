#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, itertools, configparser, numpy as np

from mpi4py import MPI
from pycwp import boxer, mio

# Define a new ConfigurationError exception
class ConfigurationError(Exception): pass

def usage(progname):
	print('USAGE: %s <configuration>' % progname, file=sys.stderr)


def makebox(config):
	'''
	Return a boxer.Box3D object configured as specified in the [grid]
	section of the provided ConfigParser object.
	'''
	try:
		# Grab the lo and hi corners of the box
		lo = [float(s) for s in config.get('grid', 'lo').split()]
		hi = [float(s) for s in config.get('grid', 'hi').split()]

		box = boxer.Box3D(lo, hi)

		# Assign the number of cells
		box.ncell = [int(s) for s in config.get('grid', 'ncells').split()]

	except (configparser.Error, ValueError):
		raise ConfigurationError('Box configuration is not valid')

	return box


def cellspeeds(cells, contrast):
	'''
	Given a set of cells, return a dictionary of relative sound-speed
	values computed from the mio.readbmat-read 3-D contrast map.
	'''
	# Convert contrast values to sound speeds
	speeds = {}
	for cell in sorted(cells, key=lambda x: (x[-1], x[-2], x[-3])):
		ct = contrast[cell[0], cell[1], cell[2]]
		speeds[cell] = 1. / np.sqrt(ct + 1.).real

	return speeds


def averager(box, seg, hits, speeds):
	'''
	Compute the average relative sound speed over the boxer.Segment3D seg.
	For each hit record of the form (i, j, k) -> (tmin, tmax) in the
	mapping hits, the relative sound speed in the interval of seg from
	length tmin to tmax is given by the dictionary entry speeds[hit[:3]].
	Outside of the hit cells, the relative sound speed is unity.
	'''
	seglen, integral = 0.0, 0.0

	for (i, j, k, tmin, tmax) in hits:
		# Grab the cell speed
		spd = speeds[(i, j, k)]
		# Grab intersection start and end lengths (as fraction of whole segment)
		tmin = max(0, tmin)
		tmax = min(1.0, tmax)
		dx = tmax - tmin
		integral += dx * spd
		seglen += dx

	# Count the default speed outside of the hit range
	integral += 1.0 - seglen
	return integral


def tracerEngine(config):
	'''
	Uses ray-marching to compute average sound speeds along segments
	passing from elements on the hemispheric transducer array to focal
	regions within a specified 3-D grid of contrast values.
	'''
	mpirank, mpisize = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
	box = makebox(config)
	if mpirank == 0:
		print('Using grid shape', box.ncell)

	# Try to read the contrast file
	try:
		contrast = mio.readbmat(config.get('grid', 'contrast'))
	except configparser.Error:
		raise ConfigurationError('Configuration must specify a contrast file')
	except: raise

	if box.ncell != contrast.shape:
		raise ConfigurationError('Configured grid does not match contrast-file grid')

	try:
		# Try to read the facet file, the list of element groups, and the group size
		elements = np.loadtxt(config.get('paths', 'elements'))
		groupRange = tuple(int(s) for s in config.get('paths', 'groups').split())
		groups = range(*groupRange)
		if mpirank == 0:
			print('Will average path speeds in range(%d,%d)' % groupRange)
		elementsPerGroup = int(config.get('paths', 'elementsPerGroup'))

		# Grab the center and output formats
		centerFormat = config.get('paths', 'centerFormat')
		outputFormat = config.get('paths', 'outputFormat')
	except configparser.Error:
		raise ConfigurationError('Incomplete [paths] section in configuration')

	for grp in groups:
		# Read the focal centers for this group
		centers = np.loadtxt(centerFormat.format(grp))
		# Compute the start and share of the local center chunk
		ncenters = len(centers)
		share, srem = ncenters // mpisize, ncenters % mpisize
		start = mpirank * share + min(mpirank, srem)
		if mpirank < srem: share += 1

		# Find the first element in this group
		first = elementsPerGroup * grp
		# Build a list to store the rank-2 average speeds for this group
		avgspeeds = []
		# Process one element at a time
		for elnum, elt in enumerate(elements[first:first+elementsPerGroup]):
			# Wait until all processes arrive
			MPI.COMM_WORLD.Barrier()

			if mpirank == 0:
				print('Ray-marching for element', elnum, 'in group', grp)

			# Build a list of segments for this group
			segs = [boxer.Segment3D(elt, cen)
					for cen in centers[start:start+share]]
			# Collect a list of hits for each focal center in this group
			hits = [box.raymarcher(seg) for seg in segs]

			# Read the sound speeds for every unique cell encountered
			speeds = cellspeeds(set(c for h in hits for c in h), contrast)

			# Average sound speed for all propagation paths
			avgs = [averager(box, seg, hl, speeds)
					for seg, hl in zip(segs, hits)]
			# Gather averages on the head node
			avgather = MPI.COMM_WORLD.gather(avgs)
			if mpirank == 0:
				avgspeeds.extend([a for avg in avgather for a in avg])

		# Save the average speeds for this group (head-node only)
		if mpirank == 0:
			np.savetxt(outputFormat.format(grp), avgspeeds, fmt='%13.8f')


if __name__ == '__main__':
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Read the configuration file
	config = configparser.SafeConfigParser()
	if len(config.read(sys.argv[1])) == 0:
		print('ERROR: configuration file %s does not exist' % sys.argv[1], file=sys.stderr)
		usage(sys.argv[0])
		sys.exit(1)

	# Call the tracer engine
	tracerEngine(config)
