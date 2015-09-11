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
	def __init__(self, servers, port, reactor=None):
		'''
		Initialize connections on the specified port to each remote
		address in the sequence servers. Invoke self.connect() to
		initialize the connections.

		If reactor is not None, capture a reference to the reactor in
		self.reactor. Otherwise, install the default Twisted reactor
		and capture a reference.
		'''
		# Initialize a map (address, port) => conductor reference
		from collections import OrderedDict
		self.conductors = OrderedDict.fromkeys(((server, port) for server in servers))
		self.port = port
		# Capture a reference to the provided reactor, or a default
		if reactor is None: from twisted.internet import reactor
		self.reactor = reactor

		self.connected = False
		self.connect()

		# XXX TODO: Require a callback to be invoked on successful connection


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

		# Join all of the deferreds into a list, storing successful results
		d = defer.gatherResults(connections, consumeErrors=True)
		d.addCallback(self.storeConductors)
		self.rootdeferred = d


	def storeConductors(self, results):
		'''
		Capture a reference to each HabisConductor proxy fetched from a
		remote connection.
		'''
		for cond, (addr, port) in zip(results, self.conductors):
			self.conductors[addr, port] = cond
		self.connected = True


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
		pfix = 'remote_'
		return dict((k.replace(pfix, '', 1) if k.startswith(pfix) else k, v)
				for k, v in self.wrapmap.iteritems())
