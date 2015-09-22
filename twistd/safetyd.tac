#!/usr/bin/env python

from twisted.protocols.basic import LineReceiver
from twisted.internet.protocol import Factory

from twisted.application import service, internet

class SafetyCommandListener(LineReceiver):
	'''
	Listen for string commands from a LabView safety measurement VI and
	fire habisc.py to control HABIS hardware.
	'''

	commands = {
			'FIRE': '/opt/habis/share/calib/calib.fire.yaml',
			'INIT': '/opt/habis/share/calib/calib.init.yaml',
			'STOP': '/opt/habis/share/calib/calib.stop.yaml',
			'QUIT': None
	}

	def __init__(self):
		self.errcount = 0


	def hangup(self, reason=''):
		'''
		Send a QUIT message and drop the connection.
		'''
		self.sendLine('QUIT,' + reason)
		self.transport.loseConnection()


	def runhabisc(self, script):
		'''
		Invoke habisc to interact with the HABIS conductor;
		raises a subprocess.CalledProcessError on error.
		'''
		from subprocess32 import check_call, CalledProcessError
		habisc = '/opt/custom-python/bin/habisc.py'
		check_call([habisc, script])


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

		# Run the desired process in the background
		from twisted.internet.threads import deferToThread
		d = deferToThread(self.runhabisc, self.commands[cmd])
		d.addCallbacks(lambda _ : self.sendLine('SUCCESS,' + cmd),
				lambda _ : self.sendLine('FAILURE,' + cmd))


# Create the twisted application object
application = service.Application("HABIS Conductor Repeater")

factory = Factory.forProtocol(SafetyCommandListener)
service = internet.TCPServer(8089, factory, interface='')
service.setServiceParent(application)
