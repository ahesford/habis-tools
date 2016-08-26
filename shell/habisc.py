#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys

import yaml

from twisted.spread import pb
from twisted.internet import reactor, defer

from habis.conductor import HabisRemoteConductorGroup as HabisRCG
from habis.conductor import HabisResponseAccumulator


def printHeader(hdr, clearline=True, stream=sys.stdout):
	'''
	Print a header followed by a line of '='.
	'''
	hdr = '| ' + hdr + ' |'
	pads = '+' + '=' * max(0, len(hdr) - 2) + '+'
	print >> stream, pads
	print >> stream, hdr
	print >> stream, pads
	if clearline: print >> stream, ''


def printResult(results, cmd):
	'''
	Pretty-print output from HabisRemoteConductorGroup.broadcast.
	'''
	if results is None: return

	acc = HabisResponseAccumulator(results)

	stdout = acc.getoutput()
	stderr = acc.getoutput(True)

	if not stdout and not stderr:
		printHeader('RUN COMMAND: %s (no output)' % (cmd.cmd,))

	if stdout:
		printHeader('RUN COMMAND: %s (stdout)' % (cmd.cmd,))
		print stdout
		print ''

	if stderr:
		printHeader('RUN COMMAND: %s (stderr)' % (cmd.cmd,))
		print stderr
		print ''

	retcode = acc.returncode()
	if retcode:
		print 'ERROR: nonzero return status %d' % retcode
		print ''

	return results


def notifyError(err, cmd):
	'''
	Print non-fatal errors to the console, simply re-raise fatal ones.
	'''
	if cmd.fatalError: raise err

	printHeader('NON-FATAL ERROR IN COMMAND: %s' % (cmd.cmd,), False)
	print err
	print ''


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
		printHeader('FATAL ERROR ENCOUNTERED (WILL TERMINATE)', False, sys.stderr)
		print >> sys.stderr, reason.value
		fatalError = True
		return reason

	# Try to find a configuration file
	confname = findconfig(sys.argv[1])

	try:
		# Execute the commands from the identified configuration file
		cseq = HabisRCG.executeCommandFile(confname,
				printResult, notifyError, sys.argv[2:], reactor)
	except Exception as e:
		print >> sys.stderr, 'Cannot process conductor script:', e
		sys.exit(1)

	# Register a fatal error
	cseq.addErrback(noteFatalError)

	# Terminate the reactor after the chain has completed
	cseq.addBoth(lambda _: reactor.stop())

	# Run the reactor to fire the commands
	reactor.run()

	# Exit nonzero if a fatal error occurred
	sys.exit(int(fatalError))
