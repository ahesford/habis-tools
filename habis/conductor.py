'''
Classes that conduct HABIS processes over a network.
'''

from twisted.spread import pb
from twisted.internet import threads, defer

class HabisConductorError(Exception): pass

class HabisRemoteConductorGroup(object):
	'''
	A client-side class for managing a collection of remote HabisConductor
	references.
	'''
	def __init__(self, servers, port,
			onConnect=None, onConnectErr=None, reactor=None):
		'''
		Initialize connections on the specified port to each remote
		address in the sequence servers. Invoke self.connect() to
		initialize the connections.

		The arguments onConnect and onConnectErr should be callables
		that accept a single argument, or else None. If not None,
		onConnect will be added as a callback to the deferred returned
		by self.connect() and be provided a reference to self.
		Likewise, onConnectErr will be added as an errback to the same
		deferred to handle connection errors. If both are defined, the
		callback and errback are installed in parallel.

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
		d = self.connect()

		if onConnect and onConnectErr:
			d.addCallbacks(onConnect, onConnectErr)
		elif onConnect:
			d.addCallback(onConnect)
		elif onConnectErr:
			d.addErrback(onConnectErr)


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


	def broadcast(self, cmd, *args, **kwargs):
		'''
		Invoke callRemote(*args, **kwargs) on each HabisConductor
		object in self.conductors.

		Returns a DeferredList joining all of the callRemote deferreds.
		'''
		calls = []
		for (addr, port), cond in self.conductors.iteritems():
			d = cond.callRemote(cmd, *args, **kwargs)
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
	def registerWrapper(cls, name, cmd):
		'''
		Create a remote_<name> method will create an instance of
		habis.wrappers.Wrapper to execute the command cmd and return
		the results of its execute() method asynchronously over the
		network. Positional arguments to the remote_<name> method are
		passed to the wrapper constructor, while keyword arguments to
		the remote_<name> method are passed to the execute() method of
		the wrapper instance.
		'''
		from types import MethodType

		# Ensure that the method does not already exist
		methodName = 'remote_' + name
		if hasattr(cls, methodName):
			raise AttributeError('Attribute %s already exists in %s' % (methodName, cls.__name__))

		docfmt = 'Instantiate a %s wrapper, asynchronously launch, and return the output'

		def callWrapper(self, *args, **kwargs):
			from habis import wrappers
			w = wrappers.CommandWrapper(cmd, *args)
			return threads.deferToThread(lambda : w.execute(**kwargs))
		callWrapper.__doc__ = docfmt % cmd

		# Add the method to the class
		setattr(cls, methodName, MethodType(callWrapper, None, cls))
		# Record the mapping between method name and wrapper object
		cls.wrapmap[methodName] = cmd


	@classmethod
	def deregisterWrapper(cls, name):
		'''
		Remove the remote_<name> method that invokes an associated
		habis.wrappers.Wrapper instance.
		'''
		methodName = 'remote_' + name
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
