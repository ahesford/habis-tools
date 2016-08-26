#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys

import yaml

from twisted.spread import pb
from twisted.internet import reactor, defer

from habis.conductor import HabisRemoteConductorGroup as HabisRCG
from habis.conductor import HabisResponseAccumulator


def printResult(results, cmd, clearline=True):
	'''
	Pretty-print output from HabisRemoteConductorGroup.broadcast.
	'''
	if results is None: return

	acc = HabisResponseAccumulator(results)
	stdout = acc.getoutput()
	if len(stdout) > 0:
		print stdout
		if clearline:
			print ''
	stderr = acc.getoutput(True)
	if len(stderr) > 0:
		print >> sys.stderr, stderr
		if clearline:
			print >> sys.stderr, ''

	retcode = acc.returncode()
	if retcode != 0:
		print >> sys.stderr, 'ERROR: nonzero return status %d' % retcode

	return results


def notifyError(err, cmd):
	'''
	Print an error encountered in a remote command invocation and, if
	cmd.fatalError is True, re-raise the error.
	'''
	if cmd.fatalError:
		print 'Fatal error in command %s: %s' % (cmd.cmd, err)
		raise err
	print 'Non-fatal error in command %s: %s' % (cmd.cmd, err)


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <cmdlist.yaml> [var=value ...]' % progname


def findconfig(confname):
	'''
	If confname contains a slash, the name contains the path; simply return
	confname.

	If confname contains no slash, return confname unmolested if any
	filesystem object exists at the location. If no object exists, return
	confname appended to a default script directory.

	Obviously, using this name introduces a potential race.
	'''
	from os.path import lexists, join

	if '/' in confname: return confname
	if lexists(confname): return confname
	return join('/opt/habis/share/habisc', confname)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Track occurrence of a fatal error
	fatalError = False
	def noteFatalError(reason):
		fatalError = True
		return reason

	# Try to find a configuration file
	confname = findconfig(sys.argv[1])

	# Execute the commands from the identified configuration file
	cseq = HabisRCG.executeCommandFile(confname,
			printResult, notifyError, sys.argv[2:], reactor)

	# Register a fatal error
	cseq.addErrback(noteFatalError)

	# Terminate the reactor after the chain has completed
	cseq.addBoth(lambda _: reactor.stop())

	# Run the reactor to fire the commands
	reactor.run()

	# Exit nonzero if a fatal error occurred
	sys.exit(int(fatalError))
