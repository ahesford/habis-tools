#!/usr/bin/env python

import sys

import yaml

from twisted.spread import pb
from twisted.internet import reactor

from habis.habiconf import HabisConfigError, HabisConfigParser

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


def printResult(results):
	'''
	Pretty-print output from multiple remote HabisConductors.
	
	If shouldStop is not False, stop the reactor after printing.
	'''
	if results is None: return

	acc = HabisResponseAccumulator(results)
	stdout = acc.getoutput()
	if len(stdout) > 0:
		print stdout
	stderr = acc.getoutput(True)
	if len(stderr) > 0:
		print >> sys.stderr, stderr

	retcode = acc.returncode()
	if retcode != 0:
		print >> sys.stderr, 'ERROR: nonzero return status %d' % retcode


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration> <cmdlist.yaml>' % progname


def configureGroup(config, cmdlist):
	csec = 'conductorClient'

	# Grab the remote host
	try:
		addrs = config.getlist(csec, 'address')
	except Exception as e:
		err = 'Configuration must specify addresses in [%s]' % csec
		raise HabisConfigError.fromException(err, e)

	# Grab the port on which to listen
	try:
		port = config.getint(csec, 'port', failfunc=lambda: 8088)
	except Exception as e:
		err = 'Invalid optional port specification in [%s]' % csec
		raise HabisConfigError.fromException(err, e)

	# Create the client-side conductor group
	hgroup = HabisRemoteConductorGroup(addrs, port, reactor)

	def remoteCaller(_):
		try: hacmd = cmdlist.pop(0)
		except IndexError: return

		if len(cmdlist) > 0:
			def nextCallback(result):
				printResult(result)
				return remoteCaller(result)
		else: 
			def nextCallback(result):
				printResult(result)
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
	if len(sys.argv) != 3:
		usage(sys.argv[0])
		sys.exit(1)

	try:
		# Read the configuration file
		config = HabisConfigParser.fromfile(sys.argv[1])
	except Exception as e:
		print >> sys.stderr, 'ERROR: could not load configuration file', sys.argv[1]
		print >> sys.stderr, '\tReason:', str(e)
		usage(sys.argv[0])
		sys.exit(1)

	try:
		# Read the command list
		commands = yaml.safe_load_all(open(sys.argv[2], 'rb'))
		cmdlist = [HabisRemoteCommand(**c) for c in commands]
	except Exception as e:
		print >> sys.stderr, 'ERROR: could not load command file', sys.argv[2]
		print >> sys.stderr, '\tReason:', str(e)
		sys.exit(1)

	# Configure the client proxy
	hgroup = configureGroup(config, cmdlist)
	reactor.run()
