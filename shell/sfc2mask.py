#!/usr/bin/env python 

import numpy as np, getopt, sys, os
from argparse import ArgumentParser

from pycwp.boxer import Box3D, Triangle3D, Octree

def usage(progname=None, fatal=False):
	if progname is None: progname = sys.argv[0]
	print('USAGE: %s <lx,ly,lz> <hx,hy,hz> <nx,ny,nz> <input> <output>' % (progname,), file=sys.stderr)
	sys.exit(fatal)


def buildtree(tris, grid):
	'''
	Given a list of Triangle3D instances tris and a Box3D grid that defines
	a voxel grid, construct an Octree to classify the indices of triangles
	in tris according to overlaps of the corresponding triangles with the
	boxes in the tree. The number of levels is chosen to be

		max(2, min(int(log2(v)) for v in grid.ncell))

	The tree is contstructed in "multibox" mode, so each triangle may be
	assigned to multiple boxes.
	'''
	# Create the tree
	nlev = max(2, min(int(np.log2(v)) for v in grid.ncell))
	otree = Octree(nlev, grid)

	# Populate and prune the tree
	def inbox(box, leaf): return tris[leaf].overlaps(box)
	otree.addleaves(range(len(tris)), inbox, True)
	otree.prune()
	return otree


def depthmap(otree, tris, grid):
	'''
	Given an Octree otree as returned by buildtree(tris, grid), a list of
	Triangle3D instances tris and a Box3D grid that defines a voxel grid,
	create a 2-D depth map ("depth") of shape grid.ncell[:2] for which
	depth[i,j] specifies the minimum index k for which the cell at
	grid.getCell(i, j, k) overlaps with a triangle in tris.

	If no cell in the column (i, j) overlaps a triangle, the value of
	depth[i,j] will be grid.ncell[2].
	'''
	ncell = grid.ncell

	# Build a grid of z-columns
	cgrid = Box3D(grid.lo, grid.hi, ncell[:2] + (1,))
	# Build the depth map
	depth = ncell[2] * np.ones(ncell[:2], dtype=np.int64)

	for i in range(ncell[0]):
		for j in range(ncell[1]):
			# Pull the column for searching
			col = cgrid.getCell(i, j, 0)
			# Search for triangles that overlap with this column
			def boxpred(box): return box.overlaps(col)
			def leafpred(leaf): return tris[leaf].overlaps(col)
			leaves = otree.search(boxpred, leafpred, { })
			# Now check each leaf against each cell in this column
			for leaf in leaves:
				tri = tris[leaf]
				# No need to check past the current low value
				for k in range(depth[i,j]):
					if tri.overlaps(grid.getCell(i, j, k)):
						# Stop checking once first hit found
						depth[i,j] = k
						break
	return depth


if __name__ == '__main__':
	parser = ArgumentParser(description='From a backscatter surface mesh, '
						'build a binary interior-exterior mask')

	parser.add_argument('-z', type=int, default=None, metavar='zmax',
				help='Exclude z slabs with indices larger than zmax')
	parser.add_argument('lo', type=float, nargs=3, help='Low corner, '
				'as "lx ly lz", of the grid spanned by the mask')
	parser.add_argument('hi', type=float, nargs=3, help='High corner, '
				'as "hx hy hz", of the grid spanned by the mask')
	parser.add_argument('ncell', type=int, nargs=3, help='Number of cells, '
				'as "nx ny nz", in the grid spanned by the mask')
	parser.add_argument('meshfile', type=str,
				help='Mesh file that defines the backscatter surface')
	parser.add_argument('maskfile', type=str, help='Mask file to be written')

	args = parser.parse_args(sys.argv[1:])

	# Read grid parameters and build the grid
	grid = Box3D(args.lo, args.hi, args.ncell)

	if args.z is not None:
		zmax = grid.ncell[2] - 1
		if not 0 < args.z < zmax:
			raise ValueError(f'Value zmax must be in range (0, {zmax})')

	# Read mesh parameters
	with np.load(args.meshfile) as mesh:
		triangles = mesh['triangles']
		nodes = mesh['nodes']

	# Build triangle list
	tris = [Triangle3D([nodes[v] for v in t]) for t in triangles]

	print('Will prepare a mask for grid', grid.lo, grid.hi, grid.ncell)

	# Build the tree
	otree = buildtree(tris, grid)
	print('Finished building Octree')

	# Build the depth map
	depth = depthmap(otree, tris, grid)
	print('Finished building depth map')

	print('Saving mask to file', args.maskfile)
	# Build the mask from the depth map
	mask = np.arange(grid.ncell[2])[np.newaxis,np.newaxis,:] >= depth[:,:,np.newaxis]

	# Squash values above zmax
	if args.z is not None: mask[:,:,args.z+1:] = False

	np.save(args.maskfile, mask)
