'''
Routines to wrap HABIS executables in Python classes.
'''

class Wrapper(object):
	'''
	A generic base class for all HABIS-executable wrappers.
	'''
	def attr2args(self):
		'''
		Convert internal attributes to a list of arguments.

		Must be defined, must return a (possibly empty) list of
		strings, and must not call super.
		'''
		raise NotImplementedError('Subclasses must override this method')


	def execute(self, *args, **kwargs):
		'''
		Invoke the command associated with the wrapper, with additional
		arguments populated by self.attr2args(). Extra arguments are
		passed to subprocess32.Popen.communicate() to control
		interaction with the child process during its lifetime.

		Returns a dictionary with keys "stdout", "stderr", and
		"returncode", where the values of "stdout" and "stderr" are the
		contents of their respective streams, and the value of
		"returncode" is an integer return code.
		'''
		from subprocess32 import Popen, PIPE

		try: cmd = self._command
		except AttributeError: cmd = None

		if not cmd:
			raise ValueError('Wrapper does not have an associated command')

		exargs = self.attr2args()

		with Popen([cmd] + exargs, stdout=PIPE,
				stderr=PIPE, universal_newlines=True) as proc:
			stdout, stderr = proc.communicate(*args, **kwargs)
			retcode = proc.returncode

		return { "stdout": stdout, "stderr": stderr, "returncode": retcode }


class Echo(Wrapper):
	'''
	A wrapper to echo arguments.
	'''
	_command = "echo"

	def __init__(self, *args, **kwargs):
		'''
		Store a string representation of all arguments for echoing.
		'''
		self.arglist = ([str(s) for s in args] + 
				["%s=%s" % kv for kv in kwargs.iteritems()])


	def attr2args(self): return self.arglist[:]


class Test256(Wrapper):
	'''
	A wrapper to control invocation of test256.
	'''
	_command = "test256"

	def __init__(self, json=True, fast=False,
			nombist=False, halt=False, script=None, board=None):
		'''
		Initialize the test256 wrapper with the given options.
		'''
		self.json = json
		self.script = script
		self.fast = fast
		self.nombist = nombist
		self.halt = halt
		self.board = board


	def attr2args(self):
		'''
		Build optional arguments for test256.
		'''
		exargs = []

		if self.json:
			exargs.append('-json')
		if self.fast:
			exargs.append('-fast')
		if self.nombist:
			exargs.append('-nombist')
		if self.halt:
			exargs.append('-halt')
		if self.board is not None:
			exargs.extend(['-board', '%d' % self.board])
		if self.script is not None:
			exargs.extend(['-script', self.script])

		return exargs
