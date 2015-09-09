'''
Classes that conduct HABIS processes over a network.
'''

from twisted.spread import pb
from twisted.internet import threads

from habis import wrappers

class HabisConductor(pb.Root):
	# Store a mapping between methods and HABIS wrappers
	wrapmap = {}

	@classmethod
	def registerWrapper(cls, name, wrapper):
		'''
		Create a remote_<name> method will invoke an instance of
		wrapper, a class like habis.wrappers.Wrapper, and return the
		results of its execute() method asynchronously over the
		network. Arguments to the remote_<name> method are passed to
		the constructor of the wrapper class.
		'''
		from types import MethodType

		# Ensure that the method does not already exist
		methodName = 'remote_' + name
		if hasattr(cls, methodName):
			raise AttributeError('Attribute %s already exists in %s' % (methodName, cls.__name__))

		docfmt = 'Instantiate a %s wrapper, asynchronously launch, and return the output'

		def callWrapper(self, *args, **kwargs):
			print 'Will invoke wrapper', name, 'with args', args, 'kwargs', kwargs
			w = wrapper(*args, **kwargs)
			d = threads.deferToThread(w.execute)
			return d
		callWrapper.__doc__ = docfmt % wrapper.__name__

		# Add the method to the class
		setattr(cls, methodName, MethodType(callWrapper, None, cls))
		# Record the mapping between method name and wrapper object
		cls.wrapmap[methodName] = wrapper


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
