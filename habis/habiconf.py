'''
Tools for manipulating and accessing HABIS configuration files.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import shlex
import yaml

from itertools import izip


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

	1. If len(files) == 1 and outpath is not a directory, return [outpath].
	2. If extension is not None, replace any existing extension in each
	   input names with the specified extension string.
	3. If outpath is not None, replace each (possibly modified) input name
	   with os.path.join(outpath, os.path.basename(input)).

	In case 1, the outpath is not checked for validity. In all other cases,
	if outpath is specified, it must refer to a valid directory. An IOError
	will be raised if this is not the case.
	'''
	from os.path import join, basename, splitext, isdir

	if len(files) == 1 and not (outpath is None or isdir(outpath)):
		return [outpath]

	# Swap the extensions if necessary
	if extension is not None:
		files = [splitext(f)[0] + '.' + extension for f in files]

	if outpath is not None:
		if not isdir(outpath):
			raise IOError('Output path must refer to an existing directory')
		files = [join(outpath, basename(f)) for f in files]

	return files


class HabisConfigError(Exception):
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


class HabisNoOptionError(HabisConfigError):
	'''
	This exception is raised when an option-get method is called, but the
	option does not exist.
	'''
	pass


class HabisNoSectionError(HabisConfigError):
	'''
	This exception is raised when an option-get method is called, but the
	section does not exist.
	'''
	pass


class HabisConfigParser(object):
	'''
	This provides convenience methods for manipulating and accessing HABIS
	configuration files with an underlying YAML representation.

	Dynamism is supported through an optional Mako preprocessor.
	'''
	def __init__(self, f=None, procincludes=True, **kwargs):
		'''
		Instantiate a configuration object with contents specified in
		f, a file-like object or string specifying a file name. If f is
		None, create an empty configuration. If procincludes is True,
		all entries in the 'include' directive of the 'general' section
		will be processed, in reverse order, to populate options not
		present in the file f.

		The file contents can include Mako template commands, which
		will be interpreted if the Mako template engine is installed.
		All extra kwargs are passed to mako.template.Template.render().
		'''
		if isinstance(f, basestring): f = open(f)

		if f is None:
			# If there is no file, just create an empty 
			self._config = { }
			return

		try:
			# Otherwise, parse the file as YAML or a Mako-templated YAML
			confbytes = f.read()
			try:
				from mako.template import Template
			except ImportError:
				if len(kwargs):
					raise TypeError('Extra keyword arguments require the Mako template engine')
				self._config = yaml.safe_load(confbytes)
			else:
				cnftmpl = Template(text=confbytes, strict_undefined=True)
				self._config = yaml.safe_load(cnftmpl.render(**kwargs))
		except Exception as e:
			err = 'Unable to parse file %s' % f.name
			raise HabisConfigError.fromException(err, e)

		# Validate the two-level structure of the configuration
		for k, v in self._config.iteritems():
			if not (hasattr(v, 'keys') and hasattr(v, 'values')):
				raise HabisConfigError('Section %s must be dictionary-like' % k)

		if not procincludes: return

		# Process the includes, if specified
		try: includes = self.getlist('general', 'include')
		except (HabisNoOptionError, HabisNoSectionError): return

		for include in reversed(includes):
			try: other = type(self)(include, True, **kwargs)
			except IOError: continue
			self.merge(other, overwrite=False)


	def merge(self, other, overwrite=False):
		'''
		Pull the sections and options from other into self, optionally
		overwriting existing options in self.
		'''
		for section in other.sections():
			# Add the section, if it does not exist
			self.add_section(section)

			for option in other.options(section):
				if not overwrite and self.has_option(section, option):
					continue
				self.set(section, option, other.get(section, option))


	def sections(self):
		'''
		Return the list of section names in the configuration.
		'''
		return self._config.keys()


	def has_section(self, section):
		'''
		Check the existence of section in the configuration.
		'''
		return section in self._config


	def has_option(self, section, option):
		'''
		Check the existing of option in the given configuration section.
		'''
		try:
			return option in self._config[section]
		except KeyError:
			raise HabisNoSectionError('The section %s does not exist' % section)


	def options(self, section):
		'''
		Return the list of options for a given section.
		'''
		try:
			return self._config[section].keys()
		except KeyError:
			raise HabisNoSectionError('The section %s does not exist' % section)


	def add_section(self, section):
		'''
		If section exists in the current configuration, do nothing.
		Otherwise, create an empty section with the specified name.
		'''
		if section not in self._config:
			self._config[section] = { }


	def set(self, section, option, value):
		'''
		Associates the given value with the given section and option.
		'''
		try:
			self._config[section][option] = value
		except KeyError:
			raise HabisNoSectionError('The section %s does not exist' % section)


	def _get_optargs(self, *args, **kwargs):
		'''
		Helper function to process the args and kwargs for "mapper",
		"default", "failfunc", and "checkmap" arguments. Returns a
		dictionary with keys corresponding to these argument names when
		specified.
		'''
		optargs = { }

		if len(args) + len(kwargs) > 4:
			raise TypeError('Total number of arguments must not exceed 4')

		try:
			optargs['mapper'] = args[0]
		except IndexError:
			try: optargs['mapper'] = kwargs.pop('mapper')
			except KeyError: pass

		try:
			optargs['default'] = args[1]
		except IndexError:
			try: optargs['default'] = kwargs.pop('default')
			except KeyError: pass

		try:
			optargs['failfunc'] = args[2]
		except IndexError:
			try: optargs['failfunc'] = kwargs.pop('failfunc')
			except KeyError: pass

		if 'failfunc' in optargs and 'default' in optargs:
			raise TypeError('Arguments "default" and "failfunc" are mutually exclusive')

		try:
			optargs['checkmap'] = args[3]
		except IndexError:
			try: optargs['checkmap'] = kwargs.pop('checkmap')
			except KeyError: pass

		if len(kwargs):
			raise TypeError('Unrecognized keyword argument %s' % kwargs.iterkeys().next())

		return optargs


	def get(self, section, option, *args, **kwargs):
		'''
		Return the value associated with the given option in the given
		section. The section must exist.

		The args and kwargs can contain the following three additional
		arguments (this order is expected for args):

		* mapper: If provided, return mapper(value) if value exists.
		* default: If provided, return default if value does not exist.
		* failfunc: If provided, call failfunc() if value does not
		  exist. Must not be specified if default is specified.
		* checkmap: If True (default) , and mapper is provided, an
		  error is raised if mapper(value) != value. If False, this
		  check is not performed.

		If a value is not found, the mapper and failfunc arguments are
		ignored regardless of any default or failfunc arguments.
		'''
		try:
			sec = self._config[section]
		except KeyError:
			raise HabisNoSectionError('The section %s does not exist' % section)

		# Process the optional arguments
		optargs = self._get_optargs(*args, **kwargs)

		try:
			val = sec[option]
		except KeyError:
			try:
				return optargs['default']
			except KeyError:
				try: return optargs['failfunc']()
				except KeyError: pass
			# Fall through to failure with no default or failfunc
			raise HabisNoOptionError('Option %s does not exist in section %s' % (option, section))

		# Return an appropriately mapped value
		try: mval = optargs['mapper'](val)
		except KeyError: return val

		# Ensure that the mapped value conforms to expectations
		if optargs.get('checkmap', True) and mval != val:
			raise TypeError('Option is not of the prescribed type')

		return mval


	def getlist(self, section, option, *args, **kwargs):
		'''
		Behaves exactly like self.get(section, option), but ensures
		that the returned value (if it exists) is a list.

		If the value stored in the given section and option is a
		string, it will be split using shlex.split() to produce the
		return list. Otherwise, the value identified by
		self.get(section, option) should be a native list.

		Optional arguments to get() are consumed by this function to
		override default behavior.

		The "mapper" and "checkmap" options behave as in get(), except
		they apply to each element in the list individually rather than
		to the list as a whole.

		The "default" and "failfunc" options behave exactly as in
		get(). These default values are not guaranteed to be lists.
		'''
		# Process optional arguments
		optargs = self._get_optargs(*args, **kwargs)

		try:
			val = self.get(section, option)
		except HabisNoOptionError as e:
			try:
				return optargs['default']
			except KeyError:
				try: return optargs['failfunc']()
				except KeyError: pass
			raise e

		if (hasattr(val, 'keys') or hasattr(val, 'values')):
			raise TypeError('Option is dictionary-like rather than list-like')

		# Split a string into a list
		if isinstance(val, basestring):
			val = shlex.split(val, comments=True)

		try: mapper = optargs['mapper']
		except KeyError: return val

		# Map the items of the list
		mval = [mapper(v) for v in val]

		# Ensure that types are right
		if (optargs.get('checkmap', True) and
				any(v != mv for v, mv in izip(val, mval))):
			raise TypeError('List items are not of the prescribed type')

		return mval


	def getrange(self, section, option, *args, **kwargs):
		'''
		Parse a value for the given section and option as either a
		range or an explicit list of integers. The value is fetched
		using self.get(section, option).

		The "mapper" and "checkmap" arguments are not supported.
		Optional arguments "default" and "failfunc" are consumed by
		this function and behave as they do with self.getlist().

		A configured range is specifed as a value

			range int1 [int2 [int3]]

		where int1, int2, and int3 are integers passed as arguments to
		range(). The ints are processed by int(), so non-integer
		arguments will be clipped instead of throwing a TypeError.

		If an explicit list of integers is desired, the value should
		take the form

			int1 [...]

		Again, int() will be used to convert non-integer arguments.

		The function shlex.split(value, comments=True) is used to parse
		the value string.
		'''
		# Process optional arguments
		optargs = self._get_optargs(*args, **kwargs)

		if 'mapper' in optargs or 'checkmap' in kwargs:
			raise TypeError("Arguments 'mapper' and 'checkmap' are not supported")

		try:
			# Grab the value explicitly as a string
			val = self.get(section, option, mapper=str)
		except HabisNoOptionError as e:
			try:
				return optargs['default']
			except KeyError:
				try: return optargs['failfunc']()
				except KeyError: pass
			raise e

		items = shlex.split(val, comments=True)
		if len(items) < 1: return []

		if items[0].lower() == 'range':
			# Process a configuration range
			if len(items) < 2 or len(items) > 4:
				raise HabisConfigError('Range specification must between 1 and 3 arguments')
			# Pass the arguments to range
			return range(*(int(i) for i in items[1:]))

		# Just convert all arguments to ints
		return [int(i) for i in items]
