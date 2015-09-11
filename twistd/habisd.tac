#!/usr/bin/env python

from twisted.spread import pb
from twisted.application import service, internet

def configureConductor():
	from habis.conductor import HabisConductor

	# The port and address on which the server will listen
	port = 8088
	addr = ''

	# The command wrappers to attach to the conductor class
	wrappers = { 
			'echo' : '/bin/echo',
			'sleep' : '/bin/sleep',
			'hostname' : '/bin/hostname',
			'uptime' : '/usr/bin/uptime',
	}

	# Register the commands to wrap
	for func, cmd in wrappers.iteritems():
		HabisConductor.registerWrapper(func, cmd)

	# Listen on the desired port and address
	factory = pb.PBServerFactory(HabisConductor())
	return internet.TCPServer(port, factory, interface=addr)

# Create the twisted application object
application = service.Application("HABIS Conductor Server")
service = configureConductor()
service.setServiceParent(application)
