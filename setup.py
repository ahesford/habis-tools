#!/usr/bin/env python
'''
HABIS: Numerical routines for processing HABIS data

The HABIS module is maintained by Andrew J. Hesford to provide useful softare
for manipulating data received from the Hemispheric Array Breast Imaging System
and for exchanging data among multiple nodes between imaging stages.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

DOCLINES = __doc__.split('\n')
VERSION = '1.1'

if __name__ == '__main__':
	try: import wheel
	except ImportError: pass

	from setuptools import setup, find_packages
	from Cython.Build import cythonize
	from glob import glob

	import numpy as np

	setup(name='habis',
			version=VERSION,
			description=DOCLINES[0],
			long_description='\n'.join(DOCLINES[2:]),
			author='Andrew J. Hesford',
			author_email='ajh@sideband.org',
			platforms=['any'], license='Closed',
			packages=['habis'],
			scripts=glob('shell/*.py'),
			ext_modules=cythonize('habis/*.pyx', 
				compiler_directives={'embedsignature': True}),
			include_dirs = [np.get_include()],
		)
