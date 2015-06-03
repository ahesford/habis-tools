'''
Tools for manipulating and accessing HABIS configuration files.
'''

import ConfigParser
import shlex

class HabisConfigError(ConfigParser.Error): 
	'''
	This generic exception is raised to identify improper HABIS configurations.
	'''
	pass


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


	def procincludes(self, section='general', option='include', **kwargs):
		'''
		Attempt to search the specified section for the named option
		which contains a list of configuration file names. These names,
		if found, will be read by this object in sequence to alter the
		configuration.

		Failure to find the section or option is silently ignored.
		Failure to parse an include file is also silently ignored.

		The kwargs are passed to self.getlist() when pulling the list
		of names.
		'''
		try:
			# Pull the list of includes as a string
			includes = self.getlist(section, option, None, **kwargs)
		except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
			# Return an empty list of nothing was found
			return []

		# Pull includes in reverse order so later files supersede earlier ones
		for include in reversed(includes):
			try:
				other = self.__class__.fromfile(include)
			except IOError:
				continue
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


	def getbooldefault(self, section, option, default):
		'''
		Try to read a Boolean value from the configuration. If the
		option is unspecified, return a provided default value. The
		section must still exist.
		'''
		try: return self.getboolean(section, option)
		except ConfigParser.NoOptionError: return default


	def getlist(self, section, option, mapper=None, **kwargs):
		'''
		Read the string value for the provided section and option, then
		return map(mapper, shlex.split(value, comments=True)).

		The kwargs are passed through to self.get()
		'''
		value = self.get(section, option, **kwargs)
		return map(mapper, shlex.split(value, comments=True))


	def getrange(self, section, option, **kwargs):
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

		The kwargs are passed through to self.get()
		'''
		value = self.get(section, option, **kwargs)
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
