'''
Routines to wrap HABIS executables in Python classes.
'''

def _strlist(a):
	'''
	If a is an iterable, convert it to a list of strings using
	str(). Otherwise, a should be an integer, and the list of
	strings is build from xrange(a).
	'''
	try: it = iter(a)
	except TypeError: it = xrange(a)
	return [str(av) for av in it]


class CommandWrapper(object):
	'''
	A generic wrapper class to execute a command with arguments.
	'''
	def __init__(self, command, *args):
		'''
		Create a wrapper to call command with the given arguments.
		'''
		if not isinstance(command, basestring):
			raise ValueError('Command must be a string')
		self.args = args
		self._command = command


	@property
	def args(self):
		'''
		Return a copy of the positional argument list.
		'''
		return list(self._args)

	@args.setter
	def args(self, a):
		'''
		Assign the positional argument list.
		'''
		self._args = [str(s) for s in a]


	@staticmethod
	def encodeResult(retcode, stdout=None, stderr=None):
		'''
		Encode the given return code and optional stdout and stderr
		strings into a dictionary response with corresponding keys
		"returncode", "stdout", "stdin".
		'''
		result = { 'returncode': retcode }
		if stdout is not None: result['stdout'] = stdout
		if stderr is not None: result['stderr'] = stderr
		return result


	@classmethod
	def _execute(cls, cmd, *args, **kwargs):
		'''
		Convenience function launch the given command with
		subprocess32.Popen, passing *args to the process and **kwargs
		to Popen.communicate(), and returning stdout, stderr, and the
		returncode as encoded by self.encodeResult()
		'''
		from subprocess32 import Popen, PIPE, TimeoutExpired

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
		to subprocess32.Popen.communicate() to control interaction with
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
		generation of additional arguments. Valid keywords are:

		- actors (default: 1): An integer or sequence of values, each
		  passed as a dynamic "actor argument" to the wrapped command
		  upon execution. A dedicated thread is spawned for each actor
		  to parallelize command execution.

		- blocks (default: 1): An integer or a sequence of arbitrary
		  values, each passed in turn as a dynamic "block argument" to
		  the wrapped command upon execution. The "actor" threads share
		  the block list.

		- chunk (default: 0): When distributing block arguments to the
		  actor threads, restrict block arguments to those that fall in
		  the listed chunk of consecutive values.

		- nchunks (default: 1): The number of chunks into which
		  consecutive values of the block list will be broken when
		  selecting the desired chunk index.

		*** For each of actors and blocks, if an integer I is specified
		    in place of a sequence, a list of range(I) is assumed.
		'''
		super(BlockCommandWrapper, self).__init__(command, *args)

		self.chunk = kwargs.pop('chunk', 0)
		self.nchunks = kwargs.pop('nchunks', 1)

		actors = kwargs.pop('actors', 1)
		blocks = kwargs.pop('blocks', 1)

		if len(kwargs) > 0:
			raise TypeError('Unexpected keyword arguments found')

		self.actors = actors
		self.blocks = blocks


	@property
	def actors(self):
		'''
		Return a copy of the actors list. When a wrapped command is
		repeatedly invoked, one thread is spawned for each actor, and
		that actor is passed as the first argument to the command.
		'''
		return list(self._actors)


	@actors.setter
	def actors(self, act): self._actors = _strlist(act)

	@property
	def blocks(self):
		'''
		Return a copy of the block list. For each invocation of the
		wrapped command, a value from the self.chunk portion of the
		block list is passed as a second argument.
		'''
		return list(self._blocks)

	@blocks.setter
	def blocks(self, blk): self._blocks = _strlist(blk)


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
			args = [actor, blk] + self.args
			try:
				result = self._execute(self._command, *args, **kwargs)
			except Exception as e:
				result = self.encodeResult(255, stderr='ERROR: Exception raised: %s' % e)

			# Add tags identifying each block
			result['actor'] = actor
			result['block'] = blk
			queue.put(result)


	def execute(self, **kwargs):
		'''
		Behaves as CommandWrapper.execute(), except that a unique
		subprocess is spawned for each actor value in self.actors.
		Values in the self.chunk portion of self.blocks are scattered
		among the actor subprocesses. Each actor and block value are
		prepended to the arguments in self.args passed to the wrapped
		command.
		'''
		from Queue import Queue, Empty
		from threading import Thread

		# Compute the local share and size
		nblocks = len(self.blocks)
		share = nblocks / self.nchunks
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

		# Store the result dictionaries pulled from the threads
		results = []
		# Testing for life BEFORE checking queue contents avoids a race
		# where threads put() results on queue and die between tests
		while any(t.is_alive() for t in workthreads) or not q.empty():
			try: result = q.get(timeout=1.0)
			except Empty: pass
			else:
				q.task_done()
				results.append(result)

		# Join all threads to clean up; should just return
		for t in workthreads: t.join()

		return results
