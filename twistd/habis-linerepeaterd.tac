#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

from twisted.application import service, internet
from twisted.internet.protocol import Factory

from habis.conductor import HabisLineRepeater

def configureRepeater():

	# The port and address on which the server will listen
	port = 8089
	addr = ''

	lrfactory = Factory.forProtocol(HabisLineRepeater)

	return internet.TCPServer(port, lrfactory, interface=addr)

# Create the twisted application object
application = service.Application("HABIS Conductor Line Repeater")
service = configureRepeater()
service.setServiceParent(application)
