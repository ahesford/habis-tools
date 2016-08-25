'''
Classes that conduct HABIS processes over a network.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

from twisted.spread import pb
from twisted.internet import threads, defer

from threading import Lock

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
		self.fatalError = fatalError

		# Copy the kwargmap dictionaries
		try: kwitems = kwargmap.iteritems()
		except AttributeError: kwitems = kwargmap.items()
		self.kwargmap = { k: dict(v) for k, v in kwitems }

		# Copy the argmap tuples or strings
		self.argmap = {}
		try:
			pwitems = argmap.iteritems()
		except AttributeError:
			pwitems = argmap.items()
		for key, args in pwitems:
			if isinstance(args, basestring):
				args = shsplit(args)
			self.argmap[key] = tuple(args)


	@property
	def isBlock(self):
		'''
		Returns True if this is a blocked command that will return a
		list of command response dictionaries for each host, False if
		this is a normal command that only returns a single command
		response dictionary.
		'''
		return self.cmd.startswith('block_')


	@classmethod
	def _updatekwargs(cls, kwargs, defaults):
		'''
		Recursively merge nested kwargs dictionaries, preferring values
		in kwargs to provided default values. The dictionary kwargs is
		mutated. Behavior is as such:

		1. If defaults[key] is not in kwargs, then

			kwargs[key] = defaults[key]

		2. If defaults[key] is in kwargs, and both defaults[key] and
		   kwargs[key] are dicts (they have "items" methods), then
		   
			kwargs[key] = _updatekwargs(kwargs[key], defaults[key])

		3. If defaults[key] is in kwargs, but at least one of
		   defaults[key] or kwargs[key] are not dicts, leave
		   kwargs[key] alone.

		Assignments in 1 (or recursively in 2) are not copied.
		'''
		for key in defaults:
			if not key in kwargs:
				kwargs[key] = defaults[key]
				continue

			kv = kwargs[key]
			dv = defaults[key]

			if hasattr(kv, 'items') and hasattr(dv, 'items'):
				cls._updatekwargs(kv, dv)


	def argsForKey(self, key):
		'''
		Return a tuple (args, kwargs) corresponding to entries of
		self.argmap[key] and self.kwargmap[key].

		Two special keys are forbidden as arguments and will be used as
		follows:

		__ALL_HOSTS__: Any entries in argmap or kwargmap that
		correspond to the requested key will be augmented with a
		corresponding entry for the special '__ALL_HOSTS__' key, if one
		exists. __ALL_HOSTS__ entries in argmap will be appended to
		key-specific entries; key-specific entries in kwargmap will
		OVERRIDE any __ALL_HOSTS__ entries when there is a conflict.

		__DEFAULT__: If the requested key is not found in argmap or
		kwargmap, the corresponding entry for special key
		'__DEFAULT__', if it exists, will be substituted.

		If no specifically requested key exists in a map, and no
		__DEFAULT__ key exists, the returned values will be () for args
		and {} for kwargs.

		*** NOTE: If __DEFAULT__ is used in place of the specifically
		requested map key, __ALL_HOSTS__ will be ignored for that map.
		'''
		ahost, dhost = '__ALL_HOSTS__', '__DEFAULT__'
		try:
			# Pull requested positional arguments with universal augment
			args = tuple(self.argmap[key] + self.argmap.get(ahost, ()))
		except KeyError:
			# Fall back to a default
			args = tuple(self.argmap.get(dhost, ()))

		try:
			# Pull the requested keyword arguments
			kwargs = dict(self.kwargmap[key])
		except KeyError:
			# Fall back to a default
			kwargs = dict(self.kwargmap.get(dhost, {}))
		else:
			# Merge requested arguments with universal augment
			self._updatekwargs(kwargs, self.kwargmap.get(ahost, {}))

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
		reference to the deepest-level underlying failure.
		'''
		from twisted.python.failure import Failure
		if isinstance(failure, Failure): failure = failure.value
		error = '%s, underlying failure: %s' % (message, failure)
		raise HabisConductorError(error)


	@staticmethod
	def _underlyingError(result):
		'''
		If result is a Python Exception instance (other than
		defer.FirstError), raise result.

		Otherwise, if result is not a Twisted Failure instance, just
		return result.

		If result is a Twisted Failure instance, recursively drill down
		the "value" attribute until an object is found that is not a
		Failure. If this is object is an Exception, raise it.
		Otherwise, return the deepest Failure that holds the object.

		As a special case, if the value of a Failure is a FirstError,
		the recursion continues by treating "value.subFailure" as the
		next value to consider. This special case is only handled once;
		if value.subFailure is itself a FirstError, then the inner
		FirstError will be raised.
		'''
		from twisted.python.failure import Failure

		if not isinstance(result, Failure):
			# Handle cases when the result is not a Failure
			if isinstance(result, Exception): raise result
			return result

		lastFailure = result
		while True:
			# Grab the value
			value = lastFailure.value

			if isinstance(value, defer.FirstError):
				# Grab underlying failures in FirstError
				value = value.subFailure

			if isinstance(value, Failure):
				# Keep drilling into deeper failures
				lastFailure = value
			elif isinstance(value, Exception):
				# Raise an Exception behind the Failure
				raise value
			else:
				# Value is not recognizable as an error
				break

		return lastFailure


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
		d.addErrback(self._underlyingError)
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

		dcall = defer.gatherResults(calls, consumeErrors=True)
		dcall.addErrback(self._underlyingError)
		return dcall


class HabisConductor(pb.Root):
	'''
	A means for wrapping commands on the server side, executing the
	commands asynchronously, and returning the output to a client.
	'''

	# Store a mapping between methods and HABIS wrappers
	wrapmap = {}

	def __init__(self,):
		'''
		Initialize the conductor with an empty map of context locks,
		and a lock to control creation of new context locks. If a
		CommandWrapper instance contains a nonempty 'context'
		attribute, a context lock will be acquired for the given
		'context' attribute value while the CommandWrapper executes.
		Context locks are created on demand and will persist for the
		life of the conductor instance.
		'''
		self._contextLocks = { }, Lock()


	def _getContextLock(self, context):
		'''
		Return (or create and return) a lock for the given named
		context. The context must be a nonempty string and is case
		insensitive. This method is serialized.
		'''
		if not (context and isinstance(context, basestring)):
			raise ValueError('Context argument must be a nonempty string')

		# Ignore case in context naming
		context = context.lower()

		# Acquire the map lock to fetch or create the lock
		with self._contextLocks[1]:
			try: return self._contextLocks[0][context]
			except KeyError:
				self._contextLocks[0][context] = Lock()
				return self._contextLocks[0][context]


	def _executeWrapper(self, wrapper, **kwargs):
		'''
		For a CommandWrapper wrapper, invoke wrapper.execute(**kwargs).
		If wrapper has a nonempty context attribute, this will be
		guarded with self._getContextLock(wrapper.context).
		'''
		# Just execute if there is no context
		if not wrapper.context: return wrapper.execute(**kwargs)
		# Otherwise, guard execution with a context lock
		with self._getContextLock(wrapper.context):
			return wrapper.execute(**kwargs)


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
			return threads.deferToThread(self._executeWrapper, w, **kwargs)

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
