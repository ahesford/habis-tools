#!/usr/bin/env python

import sys

from twisted.spread import pb

from habis.habiconf import HabisConfigError, HabisConfigParser
from habis.conductor import HabisRemoteConductorGroup

class ResponseManager(object):
	'''
	Manage responses from HABIS conductor commands.
	'''
	def __init__(self, servers, reactor=None):
		'''
		Initialize a response manager to handle errors and result
		printing coming from the provided list of servers.
		'''
		self.servers = list(servers)
		self._results = []
		if reactor is None:
			from twisted.internet import reactor
		self.reactor = reactor


	@property
	def results(self):
		'''
		A list of process result dictionaries, each of which must
		contain a 'returncode' key with an integer value, and may
		optionally contain string values for keys 'stdout' and 'stderr'
		that contain the contents of these streams.
		'''
		return list(self._results)

	@results.setter
	def results(self, rlist):
		if len(rlist) != len(self.servers):
			raise IndexError('Result and server lists must have same length')
		for result in rlist:
			if 'returncode' not in result:
				raise KeyError('Each result dictionary must have a returncode')
		self._results = list(rlist)


	def returncode(self):
		'''
		Return the first non-zero returncode in the result list,
		0 otherwise.
		'''
		if len(self.results) != len(self.servers):
			raise IndexError('Result and server lists must have same length')

		for result in self.results:
			returncode = result['returncode']
			if returncode != 0: return returncode

		return 0


	def getoutput(self, stderr=False):
		'''
		Return cleanly formatted, concatenated contents of 'stdout' (if
		stderr is False) or 'stderr' keys in the result list.
		'''
		output = ''
		key = 'stdout' if not stderr else 'stderr'
		for serv, response in zip(self.servers,self.results):
			try: text = response[key].rstrip()
			except KeyError: text = ''

			# Skip empty output
			if len(text) < 1: continue

			# Add the server name
			sserv = serv.strip()
			output += sserv + '\n' + '=' * len(sserv) + '\n'
			output += text + '\n\n'

		return output.rstrip()


	def error(self, failure):
		'''
		Print a fatal error and stop the reactor.
		'''
		print 'Fatal error:', failure.value
		reactor.stop()


def usage(progname):
	print >> sys.stderr, 'USAGE: %s <configuration>' % progname


def configureGroup(config):
	csec = 'conductorClient'

	# Grab the command to run
	try:
		command = config.getlist(csec, 'command')
	except Exception as e:
		err = 'Configuration must specify command in [%s]' % csec
		raise HabisConfigError.fromException(err, e)

	# Grab the remote host
	try:
		addrs = config.getlist(csec, 'address')
	except Exception as e:
		err = 'Configuration must specify addresses in [%s]' % csec
		raise HabisConfigError.fromException(err, e)

	# Grab the port on which to listen
	try:
		port = config.getint(csec, 'port', failfunc=lambda: 8088)
	except Exception as e:
		err = 'Invalid optional port specification in [%s]' % csec
		raise HabisConfigError.fromException(err, e)

	from twisted.internet import reactor
	responder = ResponseManager(addrs, reactor)

	def remoteCaller(_):
		d = hgroup.broadcast(*command)
		def printResult(result):
			responder.results = result
			stdout = responder.getoutput()
			if len(stdout) > 0:
				print stdout
			stderr = responder.getoutput(True)
			if len(stderr) > 0:
				print >> sys.stderr, stderr
			reactor.stop()
		d.addCallbacks(printResult, responder.error)
		return d

	# Create the client-side conductor group, run the remote command on success
	hgroup = HabisRemoteConductorGroup(addrs, port,
			remoteCaller, responder.error, responder.reactor)

	return hgroup, responder


if __name__ == "__main__":
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	try:
		# Read the configuration file
		config = HabisConfigParser.fromfile(sys.argv[1])
	except:
		print >> sys.stderr, 'ERROR: could not load configuration file %s' % sys.argv[1]
		usage(sys.argv[0])
		sys.exit(1)

	# Configure the client proxy
	hgroup, responder = configureGroup(config)
	from twisted.internet import reactor
	reactor.run()
	sys.exit(responder.returncode())
