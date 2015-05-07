'''
Tools for manipulating and accessing HABIS configuration files.
'''

import ConfigParser

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
		return the result of map(mapper, value.split()). The kwargs are
		passed through to self.get()
		'''
		value = self.get(section, option, **kwargs)
		return map(mapper, value.split())
