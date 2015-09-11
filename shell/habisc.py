#!/usr/bin/env python

import sys

from twisted.spread import pb
from twisted.internet import reactor

from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.conductor import HabisRemoteConductorGroup

class Terminator(object):
	'''
	Stops the provided reactor.
	'''
	def __init__(self):
		self.terminated = False

	def failure(self, failure):
		self.terminate('Fatal error ' + str(failure.value))

	def success(self, _):
		self.terminate('Clean reactor exit')

	def terminate(self, msg):
		if self.terminated: return

		self.terminated = True
		print msg
		reactor.stop()


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


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

	t = Terminator()

	def remoteCaller(_):
		d = hgroup.broadcast(*command)
		def printResult(result):
			print result
			reactor.stop()
		d.addCallbacks(printResult, t.failure)
		return d

	# Create the client-side conductor group, run the remote command on success
	hgroup = HabisRemoteConductorGroup(addrs, port, remoteCaller, t.failure, reactor)

	return hgroup


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
