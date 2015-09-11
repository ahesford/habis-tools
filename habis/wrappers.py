'''
Routines to wrap HABIS executables in Python classes.
'''

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


	def execute(self, **kwargs):
		'''
		Invoke the command associated with the wrapper, with additional
		arguments populated by self.args. Keyword arguments are passed
		to subprocess32.Popen.communicate() to control interaction with
		the child process during its lifetime.

		Returns an output dictionary, encoded with encodeResult(), that
		encapsulates any program output and a return code.
		'''
		from subprocess32 import Popen, PIPE, TimeoutExpired

		try: cmd = self._command
		except AttributeError: cmd = None

		if not cmd:
			raise ValueError('Wrapper does not have an associated command')

		with Popen([cmd] + self.args, stdout=PIPE, stdin=PIPE,
				stderr=PIPE, universal_newlines=True) as proc:
			try:
				stdout, stderr = proc.communicate(**kwargs)
			except TimeoutExpired:
				proc.kill()
				stdout, stderr = proc.communicate()
			retcode = proc.returncode

		return self.encodeResult(retcode, stdout, stderr)
