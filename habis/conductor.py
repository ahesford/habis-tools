'''
Classes that conduct HABIS processes over a network.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

from twisted.spread import pb
from twisted.internet import threads, defer

class HabisConductorError(Exception): pass


class HabisRemoteCommand(object):
	'''
	A class to prepare positional and keyword argument lists from a special
	configuration.
	'''
	@classmethod
	def fromargs(cls, cmd, *args, **kwargs):
		'''
		Create a HabisRemoteCommand instance for the given command,
		using the provided args and kwargs as 'default' values.
		'''
		return cls(cmd, argmap={'default': args}, kwargmap={'default': kwargs})


	def __init__(self, cmd, argmap={}, kwargmap={}, fatalError=False):
		'''
		Initialize the HabisRemoteCommand instance with the provided
		string command. Optional argmap and kwargmap should be
		dictionaries whose keys distinguish different option sets. Each
		value in kwargmap should be a kwargs dictionary. Each value in
		argmap should either be a list of positional arguments or else
		a single string; if the value is a string, it will be split
		into string arguments using shlex.split().

		The boolean value fatalError is captured as self.fatalError
		for record keeping.
		'''
		from shlex import split as shsplit

		if not isinstance(cmd, basestring):
			raise ValueError('Command must be a string')

		self.cmd = cmd
		self.kwargmap = dict(kwargmap)
		self.fatalError = fatalError

		# Build the argmap
		self.argmap = {}
		for key, args in argmap.iteritems():
			if isinstance(args, basestring):
				args = shsplit(args)
			self.argmap[key] = args


	@property
	def isBlock(self):
		'''
		Returns True if this is a blocked command that will return a
		list of command response dictionaries for each host, False if
		this is a normal command that only returns a single command
		response dictionary.
		'''
		return self.cmd.startswith('block_')


	def argsForKey(self, key):
		'''
		Return a tuple (args, kwargs) corresponding entries of
		self.argmap[key] and self.kwargmap[key]. If the key does not
		exist in either set, the 'default' key is used as a fallback.
		If the 'default' key does not exist, the argument collection
		will be empty.
		'''
		args = []
		kwargs = {}

		if key in self.argmap: args = self.argmap[key]
		elif 'default' in self.argmap: args = self.argmap['default']

		if key in self.kwargmap: kwargs = self.kwargmap[key]
		elif 'default' in self.kwargmap: kwargs = self.kwargmap['default']

		return args, kwargs


class HabisResponseAccumulator(object):
	'''
	Accumulate HABIS conductor responses from multiple hosts and distill
	the output and return codes.
	'''
	def __init__(self, responses):
		'''
		Initialize a HabisResponseAccumulator with the provided
		response lists.
		'''
		self.responses = list(responses)


	def returncode(self):
		'''
		Returns the first nonzero return code, or else 0.
		'''
		for response in self.responses:
			retcode = response['returncode']
			if retcode != 0: return retcode

		return 0


	def getoutput(self, stderr=False):
		'''
		Return distilled output from the 'stdout' or 'stderr' result
		keys if stderr is, respectively, False or True.
		'''
		output = ''
		key = 'stdout' if not stderr else 'stderr'

		for i, response in enumerate(self.responses):
			try: text = response[key].rstrip()
			except KeyError: text = ''

			# Skip empty output
			if len(text) < 1: continue

			# Grab the "host", "actor", and "block" designators
			serv = response.get('host', '')
			actor = response.get('actor', '')
			block = response.get('block', '')

			if actor:
				actor = 'Actor ' + actor
				serv = (serv + ' ' + actor) if serv else actor
			if block:
				block = 'Block ' + block
				serv = (serv + ' ' + block) if serv else block

			if not serv:
				serv = '[Missing response identifier at index %d]' % i

			output += serv + '\n' + '=' * len(serv) + '\n'
			output += text + '\n\n'

		return output.rstrip()



class HabisRemoteConductorGroup(object):
	'''
	A client-side class for managing a collection of remote HabisConductor
	references.
	'''
	def __init__(self, servers, port, reactor=None):
		'''
		Initialize HabisRemoteConductorGroup that will populate its
		conductor lists from the provided servers, each listening on
		the given port.

		If reactor is not None, a reference to will be captured in
		self.reactor. Otherwise, the default twisted.internet.reactor
		will be installed and referenced in self.reactor.
		'''
		# Initialize a map (address, port) => conductor reference
		from collections import OrderedDict
		self.conductors = OrderedDict.fromkeys(((s, port) for s in servers))
		self.port = port

		# Capture a reference to the provided reactor, or a default
		if reactor is None: from twisted.internet import reactor
		self.reactor = reactor

		self.connected = False


	def throwError(self, failure, message):
		'''
		Throw a HabisConductorError with the provided message and a
		reference to the underlying failure.
		'''
		error = '%s, underlying failure: %s' % (message, failure)
		raise HabisConductorError(error)


	def connect(self):
		'''
		Initialize connections to the servers specified as keys to
		self.conductors, adding a callback to self.storeConductors that
		stores the references in corresponding values in
		self.conductors.
		'''
		# Initialize all connections
		connections = []
		for address, port in self.conductors:
			factory = pb.PBClientFactory()
			self.reactor.connectTCP(address, port, factory)
			d = factory.getRootObject()
			d.addErrback(self.throwError, 'Failed to get root object at %s:%d' % (address, port))
			connections.append(d)

		def gatherConductors(results):
			'''
			Capture a reference to all conductors in the map
			self.conductors, then return self down the chain.
			'''
			for cond, (addr, port) in zip(results, self.conductors):
				self.conductors[addr, port] = cond
			self.connected = True
			return self

		# Join all of the deferreds into a list, storing successful results
		d = defer.gatherResults(connections, consumeErrors=True)
		d.addCallback(gatherConductors)
		return d


	def broadcast(self, hacmd):
		'''
		Invoke callRemote(cmd, *args, **kwargs) on each HabisConductor
		object in self.conductors, where cmd, args and kwargs are
		pulled from the HabisRemoteCommand instance hacmd.

		Returns a DeferredList joining all of the callRemote deferreds.
		'''
		def addhost(host):
			# Callback factory to add a host record to each response
			if hacmd.isBlock:
				# Blocked commands return a list of result dicts
				def callback(result):
					for r in result: r['host'] = host
					return result
			else:
				def callback(result):
					result['host'] = host
					return result
			return callback

		calls = []
		for (addr, port), cond in self.conductors.iteritems():
			# Try to get the args and kwargs for this server
			args, kwargs = hacmd.argsForKey(addr)
			d = cond.callRemote(hacmd.cmd, *args, **kwargs)
			d.addCallback(addhost(addr))
			d.addErrback(self.throwError, 'Remote call at %s:%d failed' % (addr, port))
			calls.append(d)
		return defer.gatherResults(calls, consumeErrors=True)


class HabisConductor(pb.Root):
	'''
	A means for wrapping commands on the server side, executing the
	commands asynchronously, and returning the output to a client.
	'''

	# Store a mapping between methods and HABIS wrappers
	wrapmap = {}

	@classmethod
	def registerWrapper(cls, name, cmd, isBlock=False):
		'''
		Create a remote_<name> (if isBlock is False) or
		remote_block_<name> (if isBlock is True) method that will
		create an instance of habis.wrappers.CommandWrapper or
		habis.wrappers.BlockCommandWrapper, respectively, to execute
		the command cmd and return the results of its execute() method
		asynchronously over the network. Positional arguments to the
		remote_<name> method are passed to the wrapper constructor,
		along with an option 'wrapinit' keyword argument (passed to the
		constructor as **wrapinit). Any other keyword arguments are
		passed to the execute() method of the wrapper instance.
		'''
		from types import MethodType

		if name.startswith('block_') and not isBlock:
			raise ValueError('Non-block command names cannot start with "block_"')

		# Ensure that the method does not already exist
		methodName = ('remote_' if not isBlock else 'remote_block_') + name

		if hasattr(cls, methodName):
			raise AttributeError('Attribute %s already exists in %s' % (methodName, cls.__name__))

		docfmt = 'Instantiate a %s wrapper, asynchronously launch, and return the output'

		if not isBlock:
			from habis.wrappers import CommandWrapper as Wrapper
		else:
			from habis.wrappers import BlockCommandWrapper as Wrapper

		def callWrapper(self, *args, **kwargs):
			wrapinit = kwargs.pop('wrapinit', {})
			w = Wrapper(cmd, *args, **wrapinit)
			return threads.deferToThread(lambda : w.execute(**kwargs))

		callWrapper.__doc__ = docfmt % cmd

		# Add the method to the class
		setattr(cls, methodName, MethodType(callWrapper, None, cls))
		# Record the mapping between method name and wrapped command
		cls.wrapmap[methodName] = { 'command': cmd, 'isBlock': isBlock }


	@classmethod
	def deregisterWrapper(cls, name, isBlock=False):
		'''
		Remove the remote_<name> or remote_block_<name> method
		(according to the truth value of isBlock) that invokes an
		associated habis.wrappers.CommandWrapper instance.
		'''
		methodName = ('remote_' if not isBlock else 'remote_block_') + name
		if methodName not in cls.wrapmap:
			raise AttributeError('Method was not previously registered with registerWrapper')

		# Delete the function and mapping
		delattr(cls, methodName)
		del cls.wrapmap[methodName]


	def remote_cmdlist(self):
		'''
		Return the class's mapping between remote methods and programs
		to execute. The 'remote_' prefix is stripped from the remote
		method names.
		'''
		from habis.wrappers import CommandWrapper
		pfix = 'remote_'
		response = dict((k.replace(pfix, '', 1) if k.startswith(pfix) else k, v)
				for k, v in self.wrapmap.iteritems())
		return CommandWrapper.encodeResult(0, str(response))
