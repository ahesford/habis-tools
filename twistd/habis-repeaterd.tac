#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

from twisted.spread import pb
from twisted.application import service, internet
from twisted.internet.protocol import Factory

from habis.conductor import HabisPerspectiveRepeater, HabisLineRepeater

def configurePerspectiveRepeater():
	# The port and address on which the server will listen
	port = 8091
	addr = ''

	# Listen on the desired port and address
	factory = pb.PBServerFactory(HabisPerspectiveRepeater())

	return internet.TCPServer(port, factory, interface=addr)

def configureLineRepeater():
	# The port and address on which the server will listen
	port = 8089
	addr = ''

	factory = Factory.forProtocol(HabisLineRepeater)

	return internet.TCPServer(port, factory, interface=addr)

# Create the twisted application object
application = service.Application("HABIS Conductor Repeater")

# Multiplex the two repeater services
mserv = service.MultiService()

# Configure each repeater service and attach to the multiservice
configureLineRepeater().setServiceParent(mserv)
configurePerspectiveRepeater().setServiceParent(mserv)

# Attach the multiservice to the application
mserv.setServiceParent(application)
