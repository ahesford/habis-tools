'''
Classes that conduct HABIS processes over a network.
'''

from twisted.spread import pb
from twisted.internet import threads

class HabisConductorProxy(object):
	'''
	A client-side proxy for a remote HabisConductor object.
	'''
	def __init__(self, factory, errback=None, *args, **kwargs):
		'''
		Instantiate a HabisConductorProxy, store a reference to the
		PBClientFactory instance factory in self.clientfactory, and
		attempt to fetch a reference to the server-side HabisConductor
		instance in the deferred self.rootdeferred.

		If errback is not None, it should be a callable to be added as
		an errback to self.rootdeferred, with *args and **kwargs
		supplied directly to the addErrback() method.
		'''
		self.clientfactory = factory
		self.rootdeferred = self.clientfactory.getRootObject()
		self.rootdeferred.addCallback(self.conductorReceived)
		if errback is not None:
			self.rootdeferred.addErrback(errback, *args, **kwargs)


	def conductorReceived(self, conductor):
		'''
		Record and return the conductor reference.
		'''
		self.conductor = conductor
		return conductor


	def callRemote(self, *args, **kwargs):
		'''
		Invoke self.conductor.callRemote(*args, **kwargs).
		'''
		return self.conductor.callRemote(*args, **kwargs)


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
		to execute.
		'''
		return self.wrapmap
