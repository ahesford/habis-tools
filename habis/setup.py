# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration
	from Cython.Build import cythonize
	from pycwp import cytools
	import os

	config = Configuration('habis', parent_package, top_path)

	cyfiles = [os.path.join(d, '*.pyx') for d in config.package_dir.values()]

	for ext in cythonize(cyfiles):
		config.ext_modules.append(ext)

	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
