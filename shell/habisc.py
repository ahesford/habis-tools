#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys

import yaml

from twisted.spread import pb
from twisted.internet import reactor, defer

from habis.conductor import HabisRemoteConductorGroup
from habis.conductor import HabisResponseAccumulator
from habis.conductor import HabisRemoteCommand


def printResult(results, clearline=True):
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


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <cmdlist.yaml> [var=value ...]' % progname


@defer.inlineCallbacks
def configureGroup(hosts, port, cmdlist):
	'''
	In a Deferred, establishes a client-side conductor group on the list of
	hosts at the given port to run the list of HabisRemoteCommand instances
	cmdlist.

	For all HabisRemoteCommand instances with a True fatalError attribute,
	any exception raised by HabisRemoteConductorGroup.broadcast will be
	raised herein, which terminates execution and fires the errback chain
	of the returned Deferred. HabisRemoteConductorGroup.connect failures
	are similarly handled.

	For all HabisRemoteCommand instances with a False fatalError attribute,
	any exception raised will be consumed and printed to the console
	without stopping execution.
	'''
	# Create the client-side conductor group
	hgroup = HabisRemoteConductorGroup(hosts, port, reactor)

	# Attempt to connect, allowing failures to fall through
	yield hgroup.connect()

	for hacmd in cmdlist:
		# Broadcast the command and wait for results
		try:
			result = yield hgroup.broadcast(hacmd)
		except Exception as e:
			# Fatal errors to fall through
			# Non-fatal errors are notify-only
			if hacmd.fatalError: raise e
			else: print 'Non-fatal error:', str(e)
		else:
			# Print the results of successful calls
			printResult(result)


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

	# Try to grab a configuration name
	confname = findconfig(sys.argv[1])

	# Parse the configuration
	try: hosts, port, cmdlist = HabisRemoteConductorGroup.parseCommandFile(confname, sys.argv[2:])
	except Exception as e:
		print >> sys.stderr, 'ERROR: could not load command file', confname
		print >> sys.stderr, 'Reason:', e
		sys.exit(1)

	# Track whether a fatal command caused premature exit
	fatalError = False

	# Configure the client proxy to run the command chain
	hgroup = configureGroup(hosts, port, cmdlist)

	# Handle fatal errors (nonfatal errors are handled internally)
	def earlyTerminator(reason):
		print 'Fatal error:', str(reason.value)
		fatalError = True
	hgroup.addErrback(earlyTerminator)

	# Terminate the reactor after the chain has completed
	hgroup.addBoth(lambda _: reactor.stop())

	# Run the reactor to fire the commands
	reactor.run()

	# Exit nonzero if a fatal error occurred
	sys.exit(int(fatalError))
