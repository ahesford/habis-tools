#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

from twisted.protocols.basic import LineReceiver
from twisted.internet.protocol import Factory

from twisted.application import service, internet

import yaml, base64

from habis.conductor import HabisRemoteConductorGroup as HabisRCG

class HabisRepeaterFactory(Factory):
	'''
	Spawn instances of HabisRepeater.
	'''
	def buildProtocol(self, addr):
		'''
		Spawn a new HabisRepeater instance to handle inbound connections.
		'''
		return HabisRepeater()


class HabisRepeater(LineReceiver):
	'''
	Listen for commands from a remote host and, in response, create a
	HabisRemoteConductorGroup to run associated remote commands on HABIS
	hardware.
	'''

	commands = {
			'FIRE': '/opt/habis/share/calib/calib.fire.yaml',
			'INIT': '/opt/habis/share/calib/calib.init.yaml',
			'STOP': '/opt/habis/share/calib/calib.stop.yaml',
			'ECHO': '/opt/habis/share/habisc/echo.yaml',
			'LASTRESULT': None,
			'QUIT': None
	}

	def __init__(self, maxerrs=3, reactor=None):
		'''
		Initialize the HabisRepeater protocol.

		The optional reactor argument is captured and passed to any
		instance of the HabisRemoteConductorGroup class used to run
		commands. This argument is otherwise unused.

		If a total of maxerrs command errors are encountered without
		an intervening successful command, the connection will be
		dropped.
		'''
		# Serialize command processing
		from threading import Lock
		self.lock = Lock()

		# Track connection errors to terminate bad clients
		self.errcount = 0
		self.maxerrs = maxerrs

		self.reactor = reactor

		# Capture results of the last HABIS command execution
		self.lastresult = []


	def sendResponse(self, msg):
		'''
		Send, using self.sendLine, the string msg after validation. If
		msg does not contain self.delimiter, it remains unchanged. If
		msg does contain self.delimiter, it will be truncated
		immediately before the first occurrence of self.delimiter prior
		to transmission.
		'''
		# Find the delimiter in the message, if it exists
		parts = msg.split(self.delimiter, 1)
		self.sendLine(parts[0] if len(parts) > 2 else msg)


	def hangup(self, reason=''):
		'''
		Send a QUIT message and drop the connection.
		'''
		self.sendResponse('QUIT,' + str(reason))
		self.transport.loseConnection()


	def sendError(self, msg):
		'''
		Send an ERR response with the given informational message (as a
		string) and increment the error counter. Terminate if the error
		count meets or exceeds self.maxerrs.

		*** NOTE: The 'ERR,' response prefix is prepended to msg
		automatically.
		'''
		self.sendResponse('ERR,' + msg)
		self.errcount += 1
		if self.errcount >= self.maxerrs:
			self.hangup('Error count too great')


	def recordCommandSuccess(self, result, cmd):
		'''
		Append to self.lastresult the tuple (cmd, result), where result
		is produced by the method HabisRemoteConductorGroup.broadcast
		for the noted cmd (an instance of HabisRemoteCommand).

		The result is expected to be a mapping from remote hostnames
		(and, for a block command, block indices) in the underlying
		HabisRemoteConductorGroup to per-host result dictionaries
		produced in accordance with CommandWrapper.encodeResult.
		'''
		self.lastresult.append((cmd.cmd, result))


	def recordCommandFailure(self, error, cmd):
		'''
		Append to self.lastresult a tuple

			(cmd, { '__ERROR__': str(error) }),

		where cmd is a HabisRemoteCommand and the error is an exception
		describing a failed HabisRemoteConductorGroup.broadcast call.

		If cmd.fatalError is True, the error will be raised after the
		append.
		'''
		self.lastresult.append((cmd.cmd, { '__ERROR__': str(error) }))
		# Signal the recorded error to the caller
		if cmd.fatalError: raise error


	def encodeLastResult(self):
		'''
		If self.lastresult is not empty, encode and return, in base64,
		a YAML representation of the list self.lastresult.

		If self.lastresult is empty, just return an empty string.

		If self.lastresult cannot be encoded with basic YAML tags, a
		descendant of yaml.YAMLError will be raised.
		'''
		if not self.lastresult: return ''
		# Encode the lastresult list or raise an error
		y = yaml.safe_dump(self.lastresult)
		return base64.b64encode(y)


	def lineReceived(self, line):
		'''
		Respond to HabisRepeater commands defined in self.commands.
		'''
		# Split the line to pull out optional arguments
		cmd = line.split(',', 1)

		if len(cmd) > 1:
			from shlex import split as shsplit
			args = shsplit(cmd[1])
		else: args = []

		cmd = cmd[0].upper()

		if cmd not in self.commands:
			# Note an error if the command is not found
			self.sendError('Command %s not recognized' % (cmd,))

		# Reset the error count on successful command
		self.errcount = 0

		if cmd == 'QUIT':
			self.hangup('Goodbye')
			return
		elif cmd == 'LASTRESULT':
			try:
				result = self.encodeLastResult()
			except yaml.YAMLError as e:
				self.sendError('Cannot encode last command result')
				print e
			else:
				# Write the length of the result string
				nresult = len(result)
				self.sendResponse('LASTRESULT,%d' % (nresult))
				if nresult:
					# Dump encoded result in "raw mode"
					self.transport.write(result)
			return

		# The lastresult list must be cleared for the to-be-run command
		self.lastresult = []

		# Lock the conductor repeater for exclusive access
		self.lock.acquire()

		try:
			# Execute appropriate command script and capture results
			cseq = HabisRCG.executeCommandFile(self.commands[cmd],
					self.recordCommandSuccess,
					self.recordCommandFailure, args, self.reactor)

			# Signal successful and failed completion to the client
			cseq.addCallbacks(lambda _: self.sendResponse('SUCCESS,' + cmd),
					lambda _: self.sendResponse('FAILURE,' + cmd))
			# Release the command lock to allow subsequent requests
			cseq.addBoth(lambda _: self.lock.release())
		except Exception as e:
			# Release the lock if the command execution failed
			# (On successful invocation, callbacks and errbacks do this)
			self.lock.release()
			# Store failed execution attempts in lastresult
			errmsg = 'Unable to process conductor script: ' + str(e)
			# There is no command associated with this failure
			self.lastresult.append((None, { '__ERROR__': errmsg }))
			# Signal failure to the caller
			self.sendResponse('FAILURE,' + cmd)

# Create the twisted application object
application = service.Application("HABIS Conductor Repeater")

service = internet.TCPServer(8089, HabisRepeaterFactory(), interface='')
service.setServiceParent(application)
