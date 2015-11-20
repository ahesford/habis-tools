#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

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
			'redistribute' : '/crypt/custom-python/bin/redistribute.py',
			'habis-remove' : '/opt/habis/bin/habis-remove',
			'habis-rescan' : '/opt/habis/bin/habis-rescan',
			'flash' : '/opt/habis/bin/habis-node-flash.sh',
			'fhfft' : '/crypt/custom-python/bin/fhfft.py',
			'rxchanlist' : '/crypt/custom-python/bin/wset-rxchanlist.py',
			'bpf' : '/crypt/custom-python/bin/wset-bpf.py',
	}

	# The block command wrappers to attach to the conductor class
	blockwrappers = {
			'echo' : '/bin/echo',
			'fieldmodel' : '/opt/habis/bin/FieldModelStepI',
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
