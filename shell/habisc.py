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

def flattener(results):
	'''
	Flatten a list-of-lists into a single list.
	'''
	return [r for result in results for r in result]


def printResult(results, isBlock=False, clearline=True):
	'''
	Pretty-print output from multiple remote HabisConductors. If isBlock is
	True, results is a list-of-lists that comes from multiple remote
	blocked commands. This list-of-lists will be flattened prior to printing.
	'''
	if results is None: return

	if isBlock: results = [r for result in results for r in result]

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
			printResult(result, hacmd.isBlock)


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

	def varpair(s):
		try: key, val = [v.strip() for v in s.split('=', 1)]
		except IndexError:
			raise ValueError('Missing equality in variable definition')
		return key, val

	varlist = dict(varpair(s) for s in sys.argv[2:])

	# Try to grab a configuration name
	confname = findconfig(sys.argv[1])

	try:
		try:
			# Mako is used for dynamic configuration is available
			from mako.template import Template
		except ImportError:
			# Without Mako, just treat the configuration as raw YAML
			print >> sys.stderr, 'WARNING: Mako template engine not found, assuming raw YAML configuration'
			configuration = yaml.safe_load(open(confname, 'rb'))
		else:
			# With Mako, render the configuration before parsing
			# Pass variable definitions from command line
			cnftmpl = Template(filename=confname, strict_undefined=True)
			configuration = yaml.safe_load(cnftmpl.render(**varlist))

		# Read connection information
		connect = configuration['connect']
		hosts = connect['hosts']
		port = connect.get('port', 8090)
		# Parse the command list
		cmdlist = [HabisRemoteCommand(**c) for c in configuration['commands']]
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
