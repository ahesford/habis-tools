#!/usr/bin/env python

from twisted.spread import pb
from twisted.application import service, internet

def configureConductor():
	from habis.conductor import HabisConductor

	# The port and address on which the server will listen
	port = 8090
	addr = ''

	# The command wrappers to attach to the conductor class
	wrappers = { 
			'echo' : '/bin/echo',
			'sleep' : '/bin/sleep',
			'hostname' : '/bin/hostname',
			'uptime' : '/usr/bin/uptime',
			'test256' : '/opt/habis/bin/test256',
			'habismon' : '/opt/habis/bin/habismon',
			'calib' : '/opt/habis/bin/calibNode',
			'redistribute' : '/opt/custom-python/bin/redistribute.py',
	}

	# The block command wrappers to attach to the conductor class
	blockwrappers = {
			'echo' : '/bin/echo'
	}

	# Register the commands to wrap
	for func, cmd in wrappers.iteritems():
		HabisConductor.registerWrapper(func, cmd, isBlock=False)
	for func, cmd in blockwrappers.iteritems():
		HabisConductor.registerWrapper(func, cmd, isBlock=True)

	# Listen on the desired port and address
	factory = pb.PBServerFactory(HabisConductor())
	return internet.TCPServer(port, factory, interface=addr)

# Create the twisted application object
application = service.Application("HABIS Conductor Server")
service = configureConductor()
service.setServiceParent(application)
