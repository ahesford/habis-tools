#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

from twisted.application import service, internet

def configureRepeater():
	from habis.conductor import HabisRepeater, HabisRepeaterFactory

	# The port and address on which the server will listen
	port = 8089
	addr = ''

	# Make sure the HabisRepeater class is aware of these commands
	commands = {
			'FIRE': '/opt/habis/share/calib/calib.fire.yaml',
			'INIT': '/opt/habis/share/calib/calib.init.yaml',
			'STOP': '/opt/habis/share/calib/calib.stop.yaml',
			'ECHO': '/opt/habis/share/habisc/echo.yaml',
	}

	for name, script in commands.iteritems():
		HabisRepeater.registerCommand(name, script)

	return internet.TCPServer(port, HabisRepeaterFactory(), interface=addr)

# Create the twisted application object
application = service.Application("HABIS Conductor Repeater")
service = configureRepeater()
service.setServiceParent(application)
