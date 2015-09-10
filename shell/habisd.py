#!/usr/bin/env python

import sys

from twisted.spread import pb
from twisted.internet import reactor

from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.conductor import HabisConductor

def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def configureConductor(config):
	csec = 'conductor'

	# Grab the commands to wrap, if possible
	try:
		wrappers = config.getlist(csec, 'wrappers',
				mapper=lambda x: x.split(':', 1))
	except Exception as e:
		err = 'Configuration must specify wrappers in [%s]' % csec
		raise HabisConfigError.fromException(err, e)

	# Grab the port on which to listen
	try:
		port = config.getint(csec, 'port', failfunc=lambda: 8088)
		addr = config.get(csec, 'address', failfunc=lambda: '')
	except Exception as e:
		err = 'Invalid optional address and port specification in [%s]' % csec
		raise HabisConfigError.fromException(err, e)

	# Register the commands to wrap
	for func, cmd in wrappers:
		HabisConductor.registerWrapper(func, cmd)

	# Listen on the desired port and address
	factory = pb.PBServerFactory(HabisConductor())
	lport = reactor.listenTCP(port, factory, interface=addr)

	return lport


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

	# Configure the server
	port = configureConductor(config)
	# Fire the reactor to launch the server
	print 'Listening on %s' % port.getHost()
	reactor.run()
