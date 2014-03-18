#!/usr/bin/env python
'''
HABIS: Numerical routines for processing HABIS data

The HABIS module is maintained by Andrew J. Hesford to provide useful softare
for manipulating data received from the Hemispheric Array Breast Imaging System
and for exchanging data among multiple nodes between imaging stages.
'''

DOCLINES = __doc__.split('\n')

def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration
	config = Configuration(None, parent_package, top_path)
	config.set_options(ignore_setup_xxx_py=True,
			assume_default_configuration=True,
			delegate_options_to_subpackages=True,
			quiet=True)
	config.add_subpackage('habis')
	config.add_scripts(['shell/*.py'])

	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup

	setup(name = 'habis', version = '0.2',
			description = DOCLINES[0],
			long_description = '\n'.join(DOCLINES[2:]),
			author = 'Andrew J. Hesford',
			author_email = 'andrew.hesford@rochester.edu',
			platforms = ['any'], license = 'BSD', packages = ['habis'],
			configuration = configuration)
