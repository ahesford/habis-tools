'''
Routines to wrap HABIS executables in Python classes.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

def _strtup(a):
	'''
	If a is an iterable, convert it to a tuple of strings using
	str(). Otherwise, a should be an integer, and the list of
	strings is built from xrange(a).

	After the tuple is produced, it is checked to ensure all entries are
	unique.
	'''
	try: it = iter(a)
	except TypeError: it = range(a)
	tup = tuple(str(av) for av in it)
	if len(set(tup)) != len(tup):
		raise ValueError('Values in sequence must be unique')
	return tup


class CommandWrapper(object):
	'''
	A generic wrapper class to execute a command with arguments.
	'''
	# Attempt to compress results with lengths longer than 4k chars
	compressLimit = 4 * 1024

	def __init__(self, command, *args, **kwargs):
		'''
		Create a wrapper to call command with the given positional
		arguments.

		Two keyword arguments, 'context', and 'contextWait', are
		supported. If provided, context must be a string or None. The
		argument is stored in the 'context' attribute of the this
		wrapper instance but is not used by the instance. If provided,
		contextWait must be a Boolean (the default value is True). This
		is stored
		'''
		if not isinstance(command, str):
			raise ValueError('Command must be a string')

		self.context = kwargs.pop('context', None)
		if not (self.context is None or isinstance(self.context, str)):
			raise ValueError('Provided context must be None or a string')

		self.contextWait = bool(kwargs.pop('contextWait', True))

		if kwargs:
			raise TypeError("Unrecognized keyword argument '%s'" % next(iter(kwargs)))

		self.args = args
		self._command = command


	@property
	def args(self):
		'''
		Return the positional argument tuple.
		'''
		return self._args

	@args.setter
	def args(self, a):
		'''
		Assign the positional argument tuple.
		'''
		self._args = tuple(str(s) for s in a)


	@classmethod
	def _compress(cls, message):
		'''
		If message has a length greater than cls.compressLimit, attempt
		to compress the message. If compression can be performed, and
		the compressed string is shorter than the input string, return
		(compressedMessage, True), where compressedMessage is the
		compressed version of the input message. Otherwise, return
		(message, False).
		'''
		try:
			if len(message) <= cls.compressLimit:
				raise ValueError('Message too short to compress')
			compressedMessage = message.encode('zlib')
			if len(compressedMessage) >= len(message):
				raise ValueError('Compression does not shorten message')
			return (compressedMessage, True)
		except Exception:
			# Return the original message if anything went wrong
			return (message, False)


	@classmethod
	def encodeResult(cls, retcode, stdout=None, stderr=None):
		'''
		Encode the given return code and optional stdout and stderr
		strings into a dictionary response with corresponding keys
		"returncode", "stdout", "stdin".

		An additional key, 'compression', holds a set that will contain
		'stdout' if stdout was compressed with stdout.encode('zlib'),
		and 'stderr' if stderr was similarly compressed.
		'''
		result = { 'returncode': retcode, 'compression': set() }

		for k, v in zip(('stdout', 'stderr'), (stdout, stderr)):
			if not v: continue
			v, compressed = cls._compress(v)
			result[k] = v
			if compressed: result['compression'].add(k)

		return result


	@staticmethod
	def decodeResult(result):
		'''
		If the input result looks like the output dictionary from
		CommandWrapper.encodeResult, undo any compression done by
		encodeResult.

		If the result contains a 'compression' key, all keys in the
		input result (except for the 'compression' key) will be copied
		into an output dictionary, but the values assigned to keys that
		are in result['compression'] will be decoded (if possible)
		using value.decode('zlib'). Undecodable values are copied
		as-is. The output will not contain a 'compression' key.

		In the case of any error (besides the copy of an undecodable
		value), the result will be returned without modification.
		'''
		try:
			compression = result['compression']

			output = { }

			for k in result:
				# Skip a 'compression' key
				if k == 'compression': continue

				v = result[k]

				if k in compression:
					# Decompress the key, if possible
					try: output[k] = v.decode('zlib')
					except Exception: output[k] = v
				else: output[k] = v

			return output
		except Exception:
			# Default to returning the untouched result
			return result


	@classmethod
	def _execute(cls, cmd, *args, **kwargs):
		'''
		Convenience function launch the given command with
		subprocess.Popen, passing *args to the process and **kwargs
		to Popen.communicate(), and returning stdout, stderr, and the
		returncode as encoded by self.encodeResult()
		'''
		from subprocess import Popen, PIPE, TimeoutExpired

		if not cmd:
			raise ValueError('Wrapper command must be defined')

		with Popen([cmd] + list(args), stdout=PIPE, stdin=PIPE,
				stderr=PIPE, universal_newlines=True) as proc:
			try:
				stdout, stderr = proc.communicate(**kwargs)
			except TimeoutExpired:
				proc.kill()
				stdout, stderr = proc.communicate()
			retcode = proc.returncode

		return cls.encodeResult(retcode, stdout, stderr)


	def execute(self, **kwargs):
		'''
		Invoke the command associated with the wrapper, with additional
		arguments populated by self.args. Keyword arguments are passed
		to subprocess.Popen.communicate() to control interaction with
		the child process during its lifetime.

		Returns an output dictionary, encoded with encodeResult(), that
		encapsulates any program output and a return code.
		'''
		return self._execute(self._command, *self.args, **kwargs)


class BlockCommandWrapper(CommandWrapper):
	'''
	A descendant of CommandWrapper that uses multiple threads to repeatedly
	invoke an executable, each time with a distinct block identifier.
	'''
	def __init__(self, command, *args, **kwargs):
		'''
		Create a wrapper to call repeatedly call the command with the
		given positional arguments *args, plus additional "actor" and
		"block" arguments that are dynamically generated and unique for
		each invocation.

		Entries in **kwargs configure work division and the dynamic
		generation of additional arguments. Valid keywords are, in
		addition to the 'context' argument that has the same meaning as
		in CommandWrapper:

		- actors (default: 1): An integer or sequence of arbitrary
		  unique values, each optionally passed as a dynamic "actor
		  argument" to the wrapped command upon execution. A dedicated
		  thread is spawned for each actor to parallelize command
		  execution.

		- blocks (default: 1): An integer or a sequence of arbitrary unique
		  values, each passed in turn as a dynamic "block argument" to
		  the wrapped command upon execution. The "actor" threads share
		  the block list.

		- chunk (default: 0): When distributing block arguments to the
		  actor threads, restrict block arguments to those that fall in
		  the listed chunk of consecutive values.

		- nchunks (default: 1): The number of chunks into which
		  consecutive values of the block list will be broken when
		  selecting the desired chunk index.

		- hideactor (default: False): If True, the "actor argument"
		  will not be passed to the wrapped command. Instead, the first
		  and only dynamic argument will be the "block argument".

		*** For each of actors and blocks, if an integer I is specified
		    in place of a sequence, a list of range(I) is assumed.
		'''
		# Initialize the underlying CommandWrapper; pull kwargs for super
		skwargs = { }
		for skw in { 'context', 'contextWait' }:
			try: skwargs[skw] = kwargs.pop(skw)
			except KeyError: pass
		super(BlockCommandWrapper, self).__init__(command, *args, **skwargs)

		self.chunk = kwargs.pop('chunk', 0)
		self.nchunks = kwargs.pop('nchunks', 1)

		actors = kwargs.pop('actors', 1)
		blocks = kwargs.pop('blocks', 1)

		self.hideactor = bool(kwargs.pop('hideactor', False))

		if kwargs:
			raise TypeError("Unexpected keyword argument '%s'" % next(iter(kwargs)))

		self.actors = actors
		self.blocks = blocks


	@property
	def actors(self):
		'''
		Return the actors tuple. When a wrapped command is repeatedly
		invoked, one thread is spawned for each actor, and that actor
		is passed as the first argument to the command.
		'''
		return self._actors


	@actors.setter
	def actors(self, act): self._actors = _strtup(act)

	@property
	def blocks(self):
		'''
		Return the block tuple. For each invocation of the wrapped
		command, a value from the self.chunk portion of the block tuple
		is passed as a second argument.
		'''
		return self._blocks

	@blocks.setter
	def blocks(self, blk): self._blocks = _strtup(blk)


	def _thread_execute(self, queue, actor, lblocks, **kwargs):
		'''
		For a given actor, invoke the wrapped command serially for each
		entry in lblocks, prepending the actor and lblocks value to
		self.args for execution. The stdout and stderr streams, along
		with the return code, for each invocation are wrapped with
		self.encodeResult() and put() into queue.

		If self._execute() raises an exception, a description of the
		error is placed on stderr, and the return code will be 255.
		'''
		for blk in lblocks:
			if self.hideactor:
				args = [blk] + list(self.args)
			else:
				args = [actor, blk] + list(self.args)

			try:
				result = self._execute(self._command, *args, **kwargs)
			except Exception as e:
				result = self.encodeResult(255, stderr='ERROR: Exception raised: %s' % e)

			# Send result and associated block back to master
			queue.put((blk, result))


	def execute(self, **kwargs):
		'''
		Behaves as CommandWrapper.execute(), except that a unique
		subprocess is spawned for each actor value in self.actors.
		Values in the self.chunk portion of self.blocks are scattered
		among the actor subprocesses. Each actor and block value are
		prepended to the arguments in self.args passed to the wrapped
		command.

		The return value is a mapping between unique block arguments and
		results of the form described in CommandWrapper.execute.

		If duplicate block indices are encountered when building the
		map, a KeyError will be raised. (This should never happen.)
		'''
		from queue import Queue, Empty
		from threading import Thread

		# Compute the local share and size
		nblocks = len(self.blocks)
		share = nblocks // self.nchunks
		rem = nblocks % self.nchunks
		start = self.chunk * share + min(self.chunk, rem)
		share += int(self.chunk < rem)

		# Compute the local blocklist
		localblocks = self.blocks[start:start+share]

		# Create a result queue and fire a thread for each actor
		workthreads = []
		q = Queue()

		nactors = len(self.actors)
		for i, actor in enumerate(self.actors):
			lb = localblocks[i::nactors]
			t = Thread(target=self._thread_execute,
					kwargs=kwargs, args = (q, actor, lb))
			t.daemon = True
			t.start()
			workthreads.append(t)

		# Store the results pulled from the threads, keyed by block
		results = { }

		# Testing for life BEFORE checking queue contents avoids a race
		# where threads put() results on queue and die between tests
		while any(t.is_alive() for t in workthreads) or not q.empty():
			try: blk, result = q.get(timeout=1.0)
			except Empty: pass
			else:
				q.task_done()
				if blk in results:
					raise KeyError('Duplicate key %d' % (blk,))
				results[blk] = result

		# Join all threads to clean up; should just return
		for t in workthreads: t.join()

		return results
