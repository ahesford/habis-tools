#!/usr/bin/env python

import sys

import yaml

from twisted.spread import pb
from twisted.internet import reactor

from habis.conductor import HabisRemoteConductorGroup
from habis.conductor import HabisResponseAccumulator
from habis.conductor import HabisRemoteCommand


def fatalError(reason):
	'''
	Print the fatal error and stop the reactor.
	'''
	print 'Fatal error:', reason.value
	reactor.stop()


def nonfatalError(reason):
	'''
	Print the non-fatal error and allow callbacks to continue.
	'''
	print 'Non-fatal error:', reason.value


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
	print >> sys.stderr, 'USAGE: %s <cmdlist.yaml>' % progname


def configureGroup(hosts, port, cmdlist):
	# Create the client-side conductor group
	hgroup = HabisRemoteConductorGroup(hosts, port, reactor)

	def remoteCaller(_):
		try: hacmd = cmdlist.pop(0)
		except IndexError: return

		# Blocked response list-of-lists must be flattened
		isBlock = hacmd.isBlock

		if len(cmdlist) > 0:
			def nextCallback(result):
				printResult(result, isBlock)
				return remoteCaller(result)
		else: 
			def nextCallback(result):
				printResult(result, isBlock, False)
				reactor.stop()

		d = hgroup.broadcast(hacmd)
		if hacmd.fatalError:
			d.addCallbacks(nextCallback, fatalError)
		else:
			d.addErrback(nonfatalError)
			d.addCallback(nextCallback)

	# Attempt to connect, running the desired command on success
	d = hgroup.connect()
	d.addCallbacks(remoteCaller, fatalError)

	return hgroup


if __name__ == "__main__":
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	try:
		try:
			# Mako is used for dynamic configuration is available
			from mako.template import Template
		except ImportError:
			# Without Mako, just treat the configuration as raw YAML
			configuration = yaml.safe_load(open(sys.argv[1], 'rb'))
		else:
			# With Mako, render the configuration before parsing
			# No rendering variables are supported
			configuration = yaml.safe_load(Template(filename=sys.argv[1]).render())

		# Read connection information
		connect = configuration['connect']
		hosts = connect['hosts']
		port = connect.get('port', 8088)
		# Parse the command list
		cmdlist = [HabisRemoteCommand(**c) for c in configuration['commands']]
	except Exception as e:
		print >> sys.stderr, 'ERROR: could not load command file', sys.argv[1]
		print >> sys.stderr, 'Reason:', e
		sys.exit(1)

	# Configure the client proxy
	hgroup = configureGroup(hosts, port, cmdlist)
	reactor.run()
