#!/usr/bin/env python

import sys

from twisted.spread import pb
from twisted.internet import reactor

from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.conductor import HabisConductorProxy

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def fatalError(failure):
	print 'Fatal error', failure
	reactor.stop()


def configureProxy(config):
	csec = 'conductorClient'

	# Grab the command to run
	try:
		command = config.getlist(csec, 'command')
	except Exception as e:
		err = 'Configuration must specify command in [%s]' % csec
		raise HabisConfigError.fromException(err, e)

	# Grab the remote host
	try:
		addr = config.get(csec, 'address')
	except Exception as e:
		err = 'Configuration must specify address in [%s]' % csec
		raise HabisConfigError.fromException(err, e)

	# Grab the port on which to listen
	try:
		port = config.getint(csec, 'port', failfunc=lambda: 8088)
	except Exception as e:
		err = 'Invalid optional port specification in [%s]' % csec
		raise HabisConfigError.fromException(err, e)


	# Listen on the desired port and address
	factory = pb.PBClientFactory()
	reactor.connectTCP(addr, port, factory)

	# Create the conductor proxy and attach to the connection
	hproxy = HabisConductorProxy(factory, errback=fatalError)
	# Configure the command to run
	hproxy.rootdeferred.addCallback(remoteCaller(hproxy, *command))

	return hproxy


def printResult(result):
	print result


def remoteCaller(hproxy, *args, **kwargs):
	def rcall(_):
		d = hproxy.callRemote(*args, **kwargs)
		d.addCallbacks(printResult, fatalError)
		d.addCallback(lambda _ : reactor.stop())
		return d
	return rcall


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
	hproxy = configureProxy(config)

	reactor.run()
