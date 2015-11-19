'''
Tools for manipulating and accessing HABIS configuration files.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import ConfigParser
import shlex


def matchfiles(files, forcematch=True):
	'''
	Given a list of files or globs, return a list of matching files.

	If forcematch is True an IOError is raised:

	1. If a specifically named file (not a glob) does not exist.
	2. If the list of matches is empty.
	'''
	import glob
	results = []
	for f in files:
		r = glob.glob(f)
		if forcematch and not (len(r) or glob.has_magic(f)):
			raise IOError("File '%s' does not exist" % f)
		results.extend(r)

	if forcematch and not len(results):
		raise IOError('No files found')

	return results


def buildpaths(files, outpath=None, extension=None):
	'''
	Given a list files of filenames, transform the names according to the
	following rules:

	1. If len(files) == 1 and outpath is not None, return [outpath].
	2. If extension is not None, replace any existing extension in each
	   input names with the specified extension string.
	3. If outpath is not None, replace each (possibly modified) input name
	   with os.path.join(outpath, os.path.basename(input)).

	In case 1, the outpath is not checked for validity. In all other cases,
	if outpath is specified, it must refer to a valid directory. An IOError
	will be raised if this is not the case.
	'''
	if len(files) == 1 and outpath is not None: return [outpath]

	from os.path import join, basename, splitext, isdir

	# Swap the extensions if necessary
	if extension is not None:
		files = [splitext(f)[0] + '.' + extension for f in files]

	if outpath is not None:
		if not isdir(outpath):
			raise IOError('Output path must refer to an existing directory')
		files = [join(outpath, basename(f)) for f in files]

	return files


class HabisConfigError(ConfigParser.Error): 
	'''
	This generic exception is raised to identify improper HABIS configurations.
	'''
	@classmethod
	def fromException(cls, msg, e):
		'''
		Create a HabisConfigError instance whose message is the
		provided string msg concatenated with a report of the
		underlying error represented by str(e).
		'''
		return cls(str(msg) + ', underlying error: ' + str(e))


class HabisConfigParser(ConfigParser.SafeConfigParser):
	'''
	This descendant of ConfigParser.SafeConfigParser provides convenience
	methods for manipulating and accessing HABIS configuration files.
	'''
	@classmethod
	def fromfile(cls, f, procincludes=True):
		'''
		Instantiate a configuration parser with contents specified in
		f, a file-like object or string specifying a file name.

		If procincludes is True, the procincludes() method is called
		with no arguments on the created instance. If this must be
		customized, pass procincludes=False and call procincludes()
		manually. The return value of procincludes() is ignored.
		'''
		# Open a named file
		if isinstance(f, basestring): f = open(f)

		# Initialize the configuration
		config = cls()
		config.readfp(f)
		if procincludes:
			config.procincludes()
		return config


	def procincludes(self, section='general', option='include', *args, **kwargs):
		'''
		Attempt to search the specified section for the named option
		which contains a list of configuration file names. These names,
		if found, will be read by this object in sequence to alter the
		configuration.

		Failure to find the section or option is silently ignored.
		Failure to parse an include file is also silently ignored.

		The args and kwargs are passed to self.getlist() when pulling
		the list of names.
		'''
		try:
			includes = self.getlist(section, option, *args, **kwargs)
		except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
			return

		# Pull includes in reverse order so later files supersede earlier ones
		for include in reversed(includes):
			try: other = self.__class__.fromfile(include)
			except IOError: continue
			self.merge(other)


	def merge(self, other, overwrite=False):
		'''
		Pull the sections and options from other into self, optionally
		overwriting existing options in self.

		DEFAULT sections from included files are ignored.

		Any interpolation is done before importing options to avoid
		potentially confusing behavior as dependencies are altered.
		'''
		for section in other.sections():
			# Add the section, if it does not exist
			try: self.add_section(section)
			except ConfigParser.DuplicateSectionError: pass

			for option in other.options(section):
				if not overwrite and self.has_option(section, option):
					continue
				self.set(section, option, other.get(section, option))


	def getboolean(self, *args, **kwargs):
		'''
		Overrid super.getboolean() to process failfunc argument as in
		self.get().

		*** NOTE ***
		The output of failfunc() is not guaranteed to be a Boolean.
		'''
		failfunc = kwargs.pop('failfunc', None)
		try:
			return ConfigParser.SafeConfigParser.getboolean(self, *args, **kwargs)
		except ConfigParser.NoOptionError: 
			if failfunc is not None: return failfunc()
			raise


	def getlist(self, *args, **kwargs):
		'''
		Read the string value for the provided section and option, then
		return map(mapper, shlex.split(value, comments=True)) for
		mapper=kwargs['mapper'] (or None if the kwarg does not exist).

		The args and kwargs are passed through to self.get()
		'''
		mapper = kwargs.pop('mapper', None)
		# Handle failfunc here to avoid processing failfunc() output
		failfunc = kwargs.pop('failfunc', None)

		try:
			value = self.get(*args, **kwargs)
		except ConfigParser.NoOptionError:
			if failfunc is not None: return failfunc()
			raise

		return map(mapper, shlex.split(value, comments=True))


	def get(self, *args, **kwargs):
		'''
		Attempt to return super.get(*args, **kwargs).

		If kwargs contains a 'failfunc' argument, and the get raises
		ConfigParser.NoOptionError, return the result of failfunc().
		'''
		failfunc = kwargs.pop('failfunc', None)

		try:
			return ConfigParser.SafeConfigParser.get(self, *args, **kwargs)
		except ConfigParser.NoOptionError:
			if failfunc is not None: return failfunc()
			raise


	def getint(self, *args, **kwargs):
		'''
		Override super.getint() to process a failfunc argument as in
		self.get(). 
		
		*** NOTE ***
		The output of failfunc() is not guaranteed to be an int.
		'''
		failfunc = kwargs.pop('failfunc', None)

		try:
			return ConfigParser.SafeConfigParser.getint(self, *args, **kwargs)
		except ConfigParser.NoOptionError:
			if failfunc is not None: return failfunc()
			raise


	def getfloat(self, *args, **kwargs):
		'''
		Override super.getfloat() to process a failfunc argument as in
		self.get().
		
		*** NOTE ***
		The output of failfunc() is not guaranteed to be a float.
		'''
		failfunc = kwargs.pop('failfunc', None)

		try:
			return ConfigParser.SafeConfigParser.getfloat(self, *args, **kwargs)
		except ConfigParser.NoOptionError:
			if failfunc is not None: return failfunc()
			raise


	def getrange(self, *args, **kwargs):
		'''
		Parse and return a configured range or explicit list of
		integers from the provided section and option.

		A configured range is specifed as a value

			range int1 [int2 [int3]]

		where int1, int2, and int3 are integers passed as arguments to
		range(). The ints are processed by int(), so non-integer
		arguments will be clipped instead of throwing a TypeError.

		The function shlex.split(value, comments=True) is used to parse
		the value string.

		The args and kwargs are passed through to self.get()
		'''
		# Process failfunc here to avoid parsing its output
		failfunc = kwargs.pop('failfunc', None)
		try:
			value = self.get(*args, **kwargs)
		except ConfigParser.NoOptionError:
			if failfunc is not None: return failfunc()
			raise

		items = shlex.split(value, comments=True)

		if len(items) < 1: return []

		if items[0].lower() == 'range':
			# Process a configuration range
			if len(items) < 2 or len(items) > 4:
				raise HabisConfigError('Range specification must between 1 and 3 arguments')
			# Pass the arguments to range
			return range(*(int(i) for i in items[1:]))

		# Just convert all arguments to ints
		return [int(i) for i in items]
