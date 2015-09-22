#!/usr/bin/env python

from twisted.protocols.basic import LineReceiver
from twisted.internet.protocol import Factory

from twisted.application import service, internet

class SafetyCommandListener(LineReceiver):
	'''
	Listen for string commands from a LabView safety measurement VI and
	fire habisc.py to control HABIS hardware.
	'''

	commands = { 'FIRE', 'INIT', 'STOP', 'QUIT' }

	def __init__(self):
		self.errcount = 0


	def hangup(self, reason=''):
		'''
		Send a QUIT message and drop the connection.
		'''
		self.sendLine('QUIT,' + reason)
		self.transport.loseConnection()


	def lineReceived(self, line):
		'''
		Respone to a "FIRE" command; return failure otherwise.
		'''
		cmd = line.upper()

		if cmd not in self.commands:
			self.sendLine('ERR,Command %s not recognized' % cmd)
			self.errcount += 1
			if self.errcount >= 3:
				self.hangup('Error count too great')
			return

		# Reset the error count on successful command
		self.errcount = 0

		if cmd == 'QUIT':
			self.hangup('Goodbye')
			return

		self.sendLine('OK,' + cmd)


# Create the twisted application object
application = service.Application("HABIS Conductor Repeater")

factory = Factory.forProtocol(SafetyCommandListener)
service = internet.TCPServer(8089, factory, interface='')
service.setServiceParent(application)
