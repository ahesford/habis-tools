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
	from setuptools import setup, find_packages, Extension
	from Cython.Build import cythonize
	from glob import glob

	import numpy as np

	ext_includes = [ np.get_include() ]

	extensions = [ Extension('*', ['habis/*.pyx'], include_dirs=ext_includes) ]

	setup(name='habis',
			version=VERSION,
			description=DOCLINES[0],
			long_description='\n'.join(DOCLINES[2:]),
			author='Andrew J. Hesford',
			author_email='ajh@sideband.org',
			platforms=['any'],
			classifiers=[
				'License :: OSI Approved :: BSD License',
				'Programming Language :: Python :: 3',
				'Intended Audience :: Developers',
				'Topic :: Scientific/Engineering',
				'Development Status :: 4 - Beta'
			],
			packages=['habis'],
			scripts=glob('shell/*.py'),
			ext_modules=cythonize(extensions,
				compiler_directives={'embedsignature': True}),
			include_dirs = ext_includes,
		)
