'''
Classes that conduct HABIS processes over a network.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

from twisted.spread import pb
from twisted.internet import threads, defer
from twisted.protocols.basic import LineReceiver
from twisted.internet.protocol import Factory

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
		Initialize a HabisResponseAccumulator with the provided mapping
		between host (and block index, if a block command was run) and
		result dictionaries.
		'''
		self.responses = dict(responses)


	def returncode(self):
		'''
		Returns the first nonzero return code, or else 0.
		'''
		for response in self.responses.itervalues():
			retcode = response['returncode']
			if retcode != 0: return retcode

		return 0


	def getoutput(self, stderr=False, hdrsep='-'):
		'''
		Return distilled output from the 'stdout' or 'stderr' result
		keys if stderr is, respectively, False or True. The host names
		(and block indices, for a block command) are printed as
		headers, separated from output by a line of repeated hdrsep
		characters.
		'''
		output = ''
		key = 'stdout' if not stderr else 'stderr'

		for hostid, response in sorted(self.responses.iteritems()):
			try: text = response[key].rstrip()
			except KeyError: text = ''

			# Skip empty output
			if len(text) < 1: continue

			# Pretty-print result key as a header
			try:
				if isinstance(hostid, basestring):
					raise ValueError('Fall-through')
				host, block = hostid
			except (TypeError, ValueError): host = str(hostid)
			else: host = str(host) + ' Block ' + str(block)

			if not host:
				host = '[Missing response identifier]'

			output += host + '\n' + hdrsep * len(host) + '\n'
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

		If reactor is not None, a reference to it will be captured in
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


	@defer.inlineCallbacks
	def runcommands(self, cmds, cb=None, eb=None):
		'''
		Invoke the sequence of HabisRemoteCommand instances in cmds, in
		order, on hosts associated with this HabisRemoteConductorGroup.
		The broadcast method is used to run each command. If this group
		is not connected, it will connect to the remote hosts before
		attempting to run the commands.

		The return value of this method is a Deferred that will fire
		after remote command execution has finished. After all remote
		commands have been completed successfully, the Deferred will
		fire its callback chain with a result of None. Early
		termination of the command sequence (because of an error) will
		case the Deferred to fire its errback chain with a Failure
		indicating the nature of the error.

		If provided, the callable cb will be called as cb(result, cmd)
		after each HabisRemoteCommand cmd completes succesfully. The
		result argument is the mapping returned by the Deferred
		provided by HabisRemoteConductorGroup.broadcast. If cb is not
		provided, the default implementation will ignore the output and
		return status of all remote commands.

		*** NOTE: A remotely executed command that returns a nonzero
		status (and, therefore, may not have run as expected) does not
		constitute a failure in HabisRemoteConductorGroup.broadcast
		and, therefore, still counts as a successful run.

		If provided, the callable eb will be called as eb(err, cmd) for
		each failed HabisRemoteCommand cmd. Within eb, consume err
		(which should be an Exception) to ignore the failure and
		continue executing the next command. Raise an Exception within
		eb to terminate early with an error.

		If eb is not provided, the default implementation ignores any
		errors raised by commands with 'fatalError' attribute values of
		False, and re-raises any error raised by commands with
		'fatalError' attribute values of True (and terminates remote
		execution thereafter).
		'''
		if not cb:
			# The default callback just ignores the result
			def cb(result, cmd): return

		if not eb:
			# The default errback fails only for fatal errors
			def eb(err, cmd):
				if not cmd.fatalError: return
				raise err

		if not self.connected:
			# Attempt to connect (failures fall through)
			yield self.connect()

			if not self.connected:
				# The attempt to connect has failed
				raise HabisConductorError('Connection attempt failed')

		# Now step through the commands in sequence
		for cmd in cmds:
			try: res = yield self.broadcast(cmd)
			except Exception as err: eb(err, cmd)
			else: cb(res, cmd)

	@classmethod
	def executeCommandFile(cls, fn, cb=None, eb=None, cvars={}, reactor=None):
		'''
		A convience method to parse a command file and run the
		discovered sequence of commands remotely. The sequence is:

		1. (hosts, port, cmds) = cls.parseCommandFile(fn, cvars)
		2. cgrp = cls(hosts, port, reactor)
		3. cgrp.runcommands(cmds, cb, eb)

		The return values from this call is the Deferred returned by
		the call to cgrp.runcommands.
		'''
		# Parse the command file and create a conductor group
		(hosts, port, cmds) = cls.parseCommandFile(fn, cvars)
		cgrp = cls(hosts, port, reactor)
		return cgrp.runcommands(cmds, cb, eb)


	@staticmethod
	def parseCommandFile(fn, cvars={}):
		'''
		Parse a file named by the string fn as a command file for a
		HabisRemoteConductorGroup. Command files are YAML documents
		that embody a dictionary with two root-level keys:

		'connect': A dictionary continaing two keys of its own that
		correspond to arguments to the HabisRemoteConductorGroup
		constructor. The first, 'hosts', provides a list of remote
		hostnames and corresponds to the 'servers' argument; the
		second, 'port',  specifies a numeric TCP port that corresponds
		to the constructor argument of the same name. The 'port' key is
		optional and will default to 8090 if unspecified.

		'commands': A list of dictionaries that can each be used as
		keyword arguments to the HabisRemoteCommand constructor to
		define a sequence of commands. That is, the i-th command for a
		HabisRemoteConductorGroup is defined by the call
		HabisRemoteCommand(**configuration['commands'][i]).

		If the Mako template engine is available, the configuration
		document is treated as a template and will first be rendered by
		Mako before being parsed as YAML. Any context variables passed
		to the renderer (as keyword arguments to the method
		mako.template.Template.render) should be provided by cvars.

		The cvars variable can have one of three forms:

		1. Dictionary-like: If **cvars can be used to unpack keyword
		   arguments, cvars is used as-is.

		2. Sequence of strings: A sequence of strings of the form
		   key=value, where the keys (context keyword variable names)
		   and values are split by the first equal sign.

		3. A single string: The string will be split using shlex.split
		   before being treated as the above-described sequence of
		   strings.

		If the Mako template engine is unavailable, the cvars argument
		is ignored.

		The return value is (hosts, port, cmdlist), where hosts and
		port are suitable for use as the first two arguments of the
		constructor for HabisRemoteConductorGroup, and cmdlist is a
		sequence of HabisRemoteCommand instances that define the
		commands to run through HabisRemoteConductorGroup.broadcast.
		'''
		from .formats import renderAndLoadYaml

		# Make sure cvars behaves as a keyword dict
		try: (lambda **r: r)(**cvars)
		except TypeError:
			# Split a single string into a sequence
			if isinstance(cvars, basestring):
				from shlex import split
				cvars = split(cvars)

			# Unpack a sequence of strings into a dictionary
			def varpair(s):
				try: key, val = [v.strip() for v in s.split('=', 1)]
				except IndexError:
					raise ValueError('Missing equality in variable definition')
				return key, val

			cvars = dict(varpair(s) for s in cvars)

		# Parse the configuration file
		configuration = renderAndLoadYaml(open(fn, 'rb').read(), **cvars)

		# Read connection information
		connect = configuration['connect']
		hosts = connect['hosts']
		port = connect.get('port', 8090)

		# Build the command list
		cmdlist = [HabisRemoteCommand(**c) for c in configuration['commands']]

		return hosts, port, cmdlist


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


	@staticmethod
	def _keybyhost(results, hosts, block=False):
		'''
		Create a map from the sequence hosts to results, a sequence of
		HabisConductor remote call responses (one per host).

		If block is False, results should be a sequence of outputs from
		CommandWrapper.execute(); the resulting map will be

			{ k: v for k, v in zip(hosts, results) }.

		If block is True, results should be a sequence of outputs from
		BlockCommandWrapper.execute(); the resulting map will be

		{ (h, k): v for h, r in zip(hosts, results)
				for k, v in results.iteritems() }.

		If duplicate keys are found, a KeyError will be raised.
		'''
		keyed = { }
		if not block:
			for k, v in zip(hosts, results):
				if k in keyed:
					raise KeyError('Duplicate key %s' % (k,))
				keyed[k] = v
		else:
			for h, r in zip(hosts, results):
				for k, v in r.iteritems():
					nk = (h, k)
					if nk in keyed:
						raise KeyError('Duplicate key %s' % (k,))
					keyed[nk] = v

		return keyed


	def broadcast(self, hacmd):
		'''
		Invoke callRemote(cmd, *args, **kwargs) on each HabisConductor
		object in self.conductors, where cmd, args and kwargs are
		pulled from the HabisRemoteCommand instance hacmd.

		Results of the remote calls are provided in a Deferred that
		will fire with a map from remote hosts (and, if hacmd.isBlock
		is True, unique block identifiers) to results encapsulated
		according to CommandWrapper.encodeResult.
		'''
		calls = []
		for (addr, port), cond in self.conductors.iteritems():
			# Try to get the args and kwargs for this server
			args, kwargs = hacmd.argsForKey(addr)
			d = cond.callRemote(hacmd.cmd, *args, **kwargs)
			d.addErrback(self.throwError, 'Remote call at %s:%d failed' % (addr, port))
			calls.append(d)

		dcall = defer.gatherResults(calls, consumeErrors=True)
		# Collapse the list of results into a single mapping
		hosts = [h[0] for h in self.conductors]
		dcall.addCallback(self._keybyhost, hosts=hosts, block=hacmd.isBlock)
		dcall.addErrback(self._underlyingError)
		return dcall


class HabisConductor(pb.Root):
	'''
	A means for wrapping commands on the server side, executing the
	commands asynchronously, and returning the output to a client.
	'''

	# Store a mapping between methods and HABIS wrappers
	wrapmap = {}

	# A set of context names for which exclusive locks can be acquired
	namedContexts = { 'gpu', 'cpu', 'mem', 'habis' }


	def __init__(self,):
		'''
		Initialize the conductor with a map of locks for exclusive
		access to all contexts in the self.namedContexts set.

		If a CommandWrapper instance contains a nonempty 'context'
		attribute, it should be a list of names in the set
		self.namedContexts for which locks will be acquired prior to
		execution.
		'''
		self._contextLocks = { k: Lock() for k in self.namedContexts }


	def _lockContexts(self, contexts, wait=True):
		'''
		Acquire, without deadlocking, all context locks in the
		collection 'contexts' in an arbitrary order, where each item in
		contexts is first stripped and converted to lowercase.

		If wait is True, this method blocks until all requested locks
		are acquired or until an error occurs. If wait is False, the
		method will raise a HabisConductorError if at least one
		requested context lock would block.
		
		A KeyError will be raised if a context name does not refer to a
		previously defined context. Any exceptions raised by
		Lock.acquire will be passed through. Any successfully acquired
		locks will be released if an error is raised.

		No value is returned is returned by this method.
		'''
		# Sleep for a random interval to avoid cyclic contentions
		import random, time

		# Keep track of the acquired locks to release on error
		acquired = [ ]

		# Clean up context names, ensure uniqueness, and pick an ordering
		contexts = list({ c.strip().lower() for c in contexts })

		while True:
			try:
				# Try to acquire locks in order
				for i, ctx in enumerate(contexts):
					try: lock = self._contextLocks[ctx]
					except KeyError:
						raise KeyError("Invalid context '%s'" % (ctx,))
					# Acquire lock, blocking for first if needed
					if lock.acquire(wait and not i):
						# Record a successful acquisition
						acquired.append(ctx)
					else:
						# Cycle blocked lock to front
						contexts = contexts[i:] + contexts[:i]
						# Silently release held locks
						self._releaseContexts(acquired, True)
						# Now no locks are held
						acquired = [ ]
						break
				else:
					# All locks were successfully acquired
					return
			except Exception as e:
				# Silently release held locks
				self._releaseContexts(acquired, True)
				raise e

			# Throw an error if waiting is not allowed
			if not wait:
				raise HabisConductorError('Context locks would block')

			# Sleep for a random time to avoid conflicts
			time.sleep(random.uniform(0., 0.1))


	def _releaseContexts(self, contexts, silent=False):
		'''
		Release all context locks in the collection 'contexts' in an
		arbitrary order, where each item in contexts is first stripped
		and converted to lowercase.

		It is the caller's responsibility to ensure that the requested
		locks were locked by the calling thread (with a preceding call
		to _lockContexts) and not some other thread; the locks will be
		released regardless.

		Any errors (invalid context names, lock issues) will be caught
		and, if silent is False, reraised by this method after all
		requested locks that can be released have been. If silent is
		True, the caught errors will be ignored.
		'''
		lastError = None
		for ctx in { c.strip().lower() for c in contexts }: 
			try:
				try: lk = self._contextLocks[ctx]
				except KeyError: raise KeyError("Invalid context '%s'" % (ctx,))
				lk.release()
			except Exception as e:
				# Capture the error and allow releases to continue
				lastError = e

		# Raise a previously captured error
		if lastError and not silent: raise lastError


	def _executeWrapper(self, wrapper):
		'''
		For a CommandWrapper wrapper, invoke wrapper.execute(). If
		wrapper has a nonempty context attribute, guard the execution
		with self._lockContexts(wrapper.context). If wrapper.context is
		a string instead of a collection, it is converted to a
		collection by calling shlex.split(wrapper.context).
		'''
		# Just execute if there is no context
		if not wrapper.context: return wrapper.execute()

		# Create a copy of the wrapper context for locking
		if isinstance(wrapper.context, basestring):
			from shlex import split
			contexts = split(wrapper.context)
		else:
			contexts = [c for c in wrapper.context]

		# Try to lock requested contexts, waiting if desired
		try: self._lockContexts(contexts, wrapper.contextWait)
		except Exception as e:
			raise HabisConductorError('Unable to lock contexts: ' + str(e))

		try:
			# Run the wrapper after contexts have been locked
			return wrapper.execute()
		finally:
			# Make sure the locked contexts are freed
			try: self._releaseContexts(contexts)
			except Exception as e:
				raise HabisConductorError('Context unlock error: ' + str(e))


	@classmethod
	def registerWrapper(cls, name, cmd, isBlock=False):
		'''
		Create a remote_<name> (if isBlock is False) or
		remote_block_<name> (if isBlock is True) method that will
		create an instance of habis.wrappers.CommandWrapper or
		habis.wrappers.BlockCommandWrapper, respectively, to execute
		the command cmd and return the results of its execute() method
		asynchronously over the network. Positional and keyword
		arguments to the remote_<name> method are passed to the wrapper
		constructor.
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
			w = Wrapper(cmd, *args, **kwargs)
			return threads.deferToThread(self._executeWrapper, w)

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

	# Configurable commands
	_commands = { }

	# System-only commands
	_syscommands = { 'LASTRESULT', 'QUIT' }

	@classmethod
	def registerCommand(cls, name, script):
		'''
		Associate the HabisRepeater command name with the given script
		for HABISRemoteConductorGroup.executeCommandFile. Names are
		always stripped and converted to uppercase.

		Raises a KeyError if the name is already registered, or an
		IOError if the specified script file does not exist.
		'''
		from os.path import isfile
		# Command name is case insensitive
		name = name.strip().upper()
		if name in cls._commands or name in cls._syscommands:
			raise KeyError("HabisRepeater command '%s' already registered" % (name,))
		if not isfile(script):
			raise IOError("HABIS command script '%s' does not exist" % (script,))
		cls._commands[name] = script


	@classmethod
	def deregisterCommand(cls, name, ignore_missing=True):
		'''
		Remove a previously registered HabisRepeater command name from
		the HabisRepeater class. Names are always stripped and
		converted to uppercase.

		Raises a KeyError if ignore_missing is False and the name was
		not previously registered. Otherwise, if the name was not
		previously registered, nothing is done.
		'''
		# Command name is case insensitive
		name = name.strip().upper()
		if name in cls._commands:
			del cls._commands[name]
		elif not ignore_missing:
			if name in cls._syscommands:
				errmsg = "Command '%s' is a system command"
			else:
				errmsg = "Unrecognized command '%s'"
			raise KeyError(errmsg % (name,))


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

		from base64 import b64encode
		from yaml import safe_dump

		# Encode the lastresult list or raise an error
		y = safe_dump(self.lastresult)
		return b64encode(y)


	def lineReceived(self, line):
		'''
		Respond to HabisRepeater commands from a client
		'''
		try: from yaml import YAMLError
		except ImportError:
			# Define a dummy YAMLError if the yaml module does not exist
			class YAMLError(Exception): pass

		# Split the line to pull out optional arguments
		cmd = line.split(',', 1)

		if len(cmd) > 1 and cmd[1]:
			# Split arguments provided to the command
			from shlex import split as shsplit
			args = shsplit(cmd[1])
		else: args = []

		cmd = cmd[0].upper()

		if cmd not in self._commands and cmd not in self._syscommands:
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
			except (YAMLError, ImportError) as e:
				self.sendError('Cannot encode last command result')
				print 'ERROR: unable to encode command results', e
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
			cseq = HabisRemoteConductorGroup.executeCommandFile(
					self._commands[cmd],
					self.recordCommandSuccess,
					self.recordCommandFailure,
					args, self.reactor)

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
