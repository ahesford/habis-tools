'''
Tools for manipulating and accessing HABIS configuration files or manipulating
program configuration.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import shlex
import contextlib

def numrange(s):
	'''
	Parse a string representing comma-separated lists of integers or ranges
	(of the form start-end) to return a sorted list of integers represented
	in the string. The start and end indices in range specifications are
	both included.

	The shlex.split(comments=True) function is used to remove whitespace
	and comments that might be embedded in the string.

	Duplicate values are OK and are coalesced in the output.
	'''
	# Use shlex to remove whitespace and comments, then split on commas
	ssec = ''.join(shlex.split(s, comments=True)).split(',')
	rvals = []

	for s in ssec:
		# Split any ranges and strip whitespace
		try: spc = [int(sv) for sv in s.split('-') if len(sv.strip())]
		except ValueError:
			raise ValueError('Invalid range component "%s"' % s)

		npc = len(spc)

		if npc == 0:
			continue
		elif npc == 1:
			rvals.append(spc[0])
		elif npc == 2:
			rseg = list(range(spc[0], spc[1] + 1))
			if not len(rseg):
				raise ValueError('Range "%s" includes no values' % s)
			rvals.extend(rseg)
		else: raise ValueError('Invalid range component "%s"' % s)

	return sorted(set(rvals))


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

	When existing extensions will be replaced with a non-None "extension"
	argument, special handling is triggered whenever the existing extension
	is ".bz2" or ".gz". In these cases, files of the form

		<basename>.<ext>.{gz,bz2}

	will have the entire "<ext>.{gz,bz2}" portion replaced. If the ".gz" or
	".bz2" extension exists without a "primary" extension, only the
	existing portion will be replaced. (In other words, the basename will
	never be removed.)
	'''
	from os.path import join, basename, splitext, isdir

	if len(files) == 1 and not (outpath is None or isdir(outpath)):
		return [outpath]

	# Swap the extensions if necessary, handling compressed extensions specially
	compext = { '.gz', '.bz2' }
	if extension is not None:
		nfiles = [ ]
		for f in files:
			bf, ef = splitext(f)
			# Look for "primary" extension in compressed files
			if ef.lower() in compext: bf = splitext(bf)[0]
			nfiles.append(bf + '.' + extension)
		files = nfiles

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


@contextlib.contextmanager
def watchConfigErrors(option, section, required=True):
	'''
	Build a context manager to watch for HabisConfigParser errors
	and, if they occur, raise a descriptive HabisConfigError.

	If required is False, a HabisNoOptionError will be ignored.
	'''
	if required: errmsg = f'Error parsing "{option}" in section "{section}"'
	else: errmsg = f'Error parsing optional "{option}" in section "{section}"'

	try:
		yield
	except Exception as e:
		if required or not isinstance(e, HabisNoOptionError):
			raise HabisConfigError.fromException(errmsg, e)


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
		if isinstance(f, str): f = open(f)

		if f is None:
			# If there is no file, just create an empty 
			self._config = { }
			return

		try:
			# Otherwise, parse the file as YAML or a Mako-templated YAML
			from .formats import renderAndLoadYaml
			confbytes = f.read()
			self._config = renderAndLoadYaml(confbytes, **kwargs)
		except Exception as e:
			err = 'Unable to parse file %s' % f.name
			raise HabisConfigError.fromException(err, e)

		# Validate the two-level structure of the configuration
		for k, v in self._config.items():
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
		return list(self._config.keys())


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
			return list(self._config[section].keys())
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
			raise TypeError('Unrecognized keyword %s' % (next(iter(kwargs.keys())),))

		return optargs


	def keys(self, section):
		'''
		Return the set of keys provided in the named section of the
		configuration.
		'''
		try:
			sec = self._config[section]
		except KeyError:
			raise HabisNoSectionError('The section %s does not exist' % section)

		return set(sec.keys())


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
		to the list as a whole. If the value is a string rather than a
		native list, the value for "checkmap" is ignored.

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
		if isinstance(val, str):
			val = shlex.split(val, comments=True)
			inferredlist = True
		else: inferredlist = False

		try: mapper = optargs['mapper']
		except KeyError: return val

		# Map the items of the list
		mval = [mapper(v) for v in val]

		# Ensure that types are right
		if (not inferredlist and optargs.get('checkmap', True) and
				any(v != mv for v, mv in zip(val, mval))):
			raise TypeError('List items are not of the prescribed type')

		return mval


	def getrange(self, section, option, *args, **kwargs):
		'''
		Parse a value self.get(section, option) as a list of integers
		or ranges. The return value is an expanded and sorted list of
		integers.

		The "mapper" and "checkmap" arguments are not supported.
		Optional arguments "default" and "failfunc" are consumed by
		this function and behave as they do with self.getlist().

		Ranges take the form

			2, 5, 7-12, 15

		Range specifications of the form "start-end" include both the
		start and end indices.
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

		return numrange(val)
