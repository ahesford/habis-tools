#!/usr/bin/env python

import os, sys, getopt, random

from habis import formats

def hostislocal(host):
	'''
	Returns True if the provided hostname or IP address belongs to the
	local machine, False otherwise. Works by trying to bind to the address.
	'''
	import socket, errno
	addr = socket.gethostbyname(host)
	s = socket.socket()

	try:
		try: s.bind((addr, 0))
		except socket.error as err:
			if err.args[0] == errno.EADDRNOTAVAIL: return False
			else: raise
	finally: s.close()

	return True


def xfer(srcfile, dstfile, host):
	'''
	If the transfer is local (host is None or ''), attempt to hard-link the
	source file srcfile at the location dstfile. If the link fails, attempt
	a copy. Returns True (False) if the file was linked (copied), or raises
	subprocess.CalledProcessError on error.

	If the transfer is not local, transfer srcfile to <host>:<dstfile> with
	scp. Returns True Unless a subprocess.CalledProcessError is raised.
	'''
	from subprocess32 import check_call, CalledProcessError

	if host:
		check_call(['scp', srcfile, host + ':' + dstfile])
		return True

	if os.path.abspath(srcfile) == os.path.abspath(dstfile):
		# If source and destination are the same, no need to link
		return True

	try:
		check_call(['ln', srcfile, dstfile])
		return True
	except CalledProcessError:
		check_call(['cp', srcfile, dstfile])
		return False


def usage(progname = 'redistribute.py'):
	binfile = os.path.basename(progname)
	print >> sys.stderr, "Usage:", binfile, "[-h] [-z] -g <grpcounts> -n <hosts> -a <num> <srcdir>/<inprefix> <destdir>/<outprefix>"
	print >> sys.stderr, "\t-g <grpcounts>: A comma-separated list of file groups ngrp1,ngrp2,[...]"
	print >> sys.stderr, "\t-n <hosts>: A comma-separated list of of hosts participating in the transfer"
	print >> sys.stderr, "\t-a <num>: Accumulate all files with the same value for group <num> on a given host"
	print >> sys.stderr, "\t-z: Zero-pad group indices in destination file names"


if __name__ == '__main__':
	optlist, args = getopt.getopt(sys.argv[1:], 'hzg:n:a:')

	hostlist = None
	grpcounts = None
	groupacc = None
	zeropad = False

	# Parse the option list
	for opt in optlist:
		if opt[0] == '-n':
			hostlist = [(o, hostislocal(o)) for o in opt[1].split(',')]
		elif opt[0] == '-g':
			grpcounts = [int(o) for o in opt[1].split(',')]
		elif opt[0] == '-a':
			groupacc = int(opt[1])
		elif opt[0] == '-z':
			zeropad = True
		elif opt[0] == '-h':
			usage(sys.argv[0])
			sys.exit()
		else:
			usage(sys.argv[0])
			sys.exit('Invalid argument')

	# Ensure mandatory arguments were provided
	if len(args) < 2 or not (hostlist or grpcounts) or groupacc is None:
		usage(sys.argv[0])
		sys.exit('Improper argument specification')

	ngroups = len(grpcounts)
	if groupacc < 0 or groupacc >= ngroups:
		sys.exit('Option to -a must be positive and less than number of groups')
	
	# Grab the in prefix and the source directory
	srcdir, inprefix = os.path.split(args[0])
	# The destination directory and prefix can remain joined together
	destform = args[1]

	print 'Transfer from %s to %s' % (srcdir, os.path.dirname(destform))

	# Figure out the start and share of each destination host
	nhosts = len(hostlist)
	share, rem = grpcounts[groupacc] / nhosts, grpcounts[groupacc] % nhosts
	dstshares = [(i * share + min(i, rem), share + int(i < rem)) for i in range(nhosts)]
	# Map group indices to hosts
	def grouptohost(i):
		for j, (start, share) in enumerate(dstshares):
			if start <= i < start + share:
				return hostlist[j]
		raise IndexError('Group index is not assigned to a host')

	# Build a randomly-ordered list of all local files
	locfiles = formats.findenumfiles(srcdir, inprefix, '\.dat', ngroups)
	random.shuffle(locfiles)

	if zeropad:
		from pycwp.util import zeropad as zpadstr

	# Transfer each file one-by-one
	for lfile in locfiles:
		srcfile, indices = lfile[0], lfile[1:]
		host, isLocal = grouptohost(indices[groupacc])
		# Build a string representation of the group indices
		if zeropad:
			grpstr = '-'.join(zpadstr(d, m) for d, m in zip(indices, grpcounts))
		else:
			grpstr = '-'.join(str(d) for d in indices)

		# Build the destination file name
		destfile = destform + '-' + grpstr + '.dat'
		# Perform the transfers
		xfer(srcfile, destfile, host if not isLocal else None)

	print 'Finished data exchange'
