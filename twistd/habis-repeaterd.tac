#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

from twisted.spread import pb
from twisted.application import service, internet

from habis.conductor import HabisPerspectiveRepeater

def configureRepeater():

	# The port and address on which the server will listen
	port = 8091
	addr = ''

	# Listen on the desired port and address
	factory = pb.PBServerFactory(HabisPerspectiveRepeater())

	return internet.TCPServer(port, factory, interface=addr)

# Create the twisted application object
application = service.Application("HABIS Conductor Perspective Repeater")
service = configureRepeater()
service.setServiceParent(application)
