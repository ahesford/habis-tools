'''
Classes that conduct HABIS processes over a network.
'''

from twisted.spread import pb
from twisted.internet import threads

class HabisConductorProxy(object):
	'''
	A client-side proxy for a remote HabisConductor object.
	'''
	def __init__(self):
		'''
		Instantiate a HabisConductorProxy with an empty proxymap.
		'''
		self.proxymap = {}

	def conductorReceived(self, conductor):
		'''
		Record the conductor reference and discover remote methods to
		populate the proxy map.

		Returns the deferred that will fire when the map of remote
		methods is available. A callback to self.buildProxyMap is
		attached to the deferred.
		
		No errbacks are attached to the deferred.
		'''
		self.conductor = conductor
		d = self.conductor.callRemote("cmdlist")
		d.addCallback(self.buildProxyMap)
		return d


	def buildProxyMap(self, cmdmap):
		'''
		Record the list of remote methods on the associated conductor
		in the dictionary self.proxymap. The argument cmdmap, returned
		from the HabisConductor remote_cmdlist method, is a dictionary
		whose keys are remote_<name> methods and whose values are the
		remote executables to be called on the conductor server. The
		local proxymap takes the same form, except the leading
		"remote_" is stripped.

		Returns self.proxymap after construction to allow callback
		chaining.
		'''
		proxymap = {}
		for func, cmd in cmdmap.iteritems():
			if not func.startswith("remote_"):
				raise ValueError("Remote callables must start with 'remote_'")
			lfunc = func.replace("remote_", "", 1)
			proxymap[lfunc] = cmd
		self.proxymap = proxymap
		return self.proxymap


	def callRemote(self, cmd, *args, **kwargs):
		'''
		If the requested cmd is in the keys of self.proxymap, forward
		the requested call to the conductor associated with the
		instance and return the resulting deferred.

		Otherwise, raise a KeyError.
		'''
		if cmd not in self.proxymap:
			raise KeyError('Command is not in proxymap for conductor')

		return self.conductor.callRemote(cmd, *args, **kwargs)


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
