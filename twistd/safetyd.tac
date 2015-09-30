#!/usr/bin/env python

from twisted.protocols.basic import LineReceiver
from twisted.internet.protocol import Factory

from twisted.application import service, internet

class SafetyCommandFactory(Factory):
	'''
	Spawn instances of SafetyCommandListener bound by a common
	serialization lock.
	'''
	def __init__(self):
		from threading import Lock
		self.lock = Lock()


	def buildProtocol(self, addr):
		'''
		Spawn a new SafetyCommandListener with a reference to the
		global lock.
		'''
		return SafetyCommandListener(lock=self.lock)


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

	def __init__(self, lock=None):
		'''
		Initialize the protocol with an optional mutex for
		serialization of habisc commands.
		'''
		self.errcount = 0
		self.lock = lock


	def hangup(self, reason=''):
		'''
		Send a QUIT message and drop the connection.
		'''
		self.sendLine('QUIT,' + reason)
		self.transport.loseConnection()


	def runhabisc(self, script, args=None):
		'''
		Invoke habisc to interact with the HABIS conductor;
		raises a subprocess.CalledProcessError on error.

		If args is not None, it should be an extra list of string
		arguments that will be appended to the habisc command line.
		'''
		from subprocess32 import check_call, CalledProcessError
		habisc = '/opt/custom-python/bin/habisc.py'

		try: self.lock.acquire()
		except AttributeError: pass

		try:
			check_call([habisc, script] + (args or []))
		finally:
			try: self.lock.release()
			except AttributeError: pass


	def lineReceived(self, line):
		'''
		Respond to safetyd commands defined in self.commands.
		'''
		# Split the line to pull out optional arguments
		cmd = line.split(',', 1)

		if len(cmd) > 1:
			from shlex import split as shsplit
			args = shsplit(cmd[1])
		else: args = []

		cmd = cmd[0].upper()

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
		d = deferToThread(self.runhabisc, self.commands[cmd], args)
		d.addCallbacks(lambda _ : self.sendLine('SUCCESS,' + cmd),
				lambda _ : self.sendLine('FAILURE,' + cmd))


# Create the twisted application object
application = service.Application("HABIS Conductor Repeater")

service = internet.TCPServer(8089, SafetyCommandFactory(), interface='')
service.setServiceParent(application)
