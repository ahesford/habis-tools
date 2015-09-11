#!/usr/bin/env python

import sys

from twisted.spread import pb
from twisted.internet import reactor

from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.conductor import HabisRemoteConductorGroup

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def fatalError(failure):
	print 'Fatal error', failure.value
	reactor.stop()


def configureGroup(config):
	csec = 'conductorClient'

	# Grab the command to run
	try:
		command = config.getlist(csec, 'command')
	except Exception as e:
		err = 'Configuration must specify command in [%s]' % csec
		raise HabisConfigError.fromException(err, e)

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

	def remoteCaller(_):
		d = hgroup.broadcast(*command)
		d.addCallbacks(printResult, fatalError)
		def stopper(x):
			print 'Stopper', x
			reactor.stop()
		d.addCallback(lambda _ : reactor.stop())
		return d

	# Create the client-side conductor group, run the remote command on success
	hgroup = HabisRemoteConductorGroup(addrs, port, remoteCaller, fatalError, reactor)

	return hgroup


def printResult(result):
	print result


if __name__ == "__main__":
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	try:
		# Read the configuration file
		config = HabisConfigParser.fromfile(sys.argv[1])
	except:
		print >> sys.stderr, 'ERROR: could not load configuration file %s' % sys.argv[1]
		usage(sys.argv[0])
		sys.exit(1)

	# Configure the client proxy
	hgroup = configureGroup(config)
	reactor.run()
