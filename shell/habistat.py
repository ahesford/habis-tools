#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import os
import uptime
import yaml
import glob


from collections import OrderedDict

def habistat():
	'''
	Return, as a YAML block-formatted string, the system uptime, the load
	average, and a list of identified HABIS devices.
	'''
	try:
		utime = uptime.uptime()
		if not utime: raise ValueError
	except Exception as e:
		utime = 'Indeterminate'
	else:
		tstr = ''
		days, utime = utime // 86400, utime % 86400
		if days: tstr += '%dd' % (days,)

		hours, utime = utime // 3600, utime % 3600
		if hours: tstr += '%dh' % (hours,)

		minutes, utime = utime // 60, utime % 60
		if minutes: tstr += '%dm' % (minutes,)

		if utime or not tstr: tstr += '%0.2fs' % (utime,)

		utime = tstr

	try: loadavgs = os.getloadavg()
	except Exception as e: loadavgs = 'Indeterminate'
	else: loadavgs = dict(zip((1, 5, 15), loadavgs))

	hadevs = sorted(glob.glob('/dev/habis[0-9]*'))

	stats = { 'Uptime': utime, 'Load averages': loadavgs, 'HABIS devices': hadevs }

	return yaml.safe_dump(stats, default_flow_style=False)


if __name__ == '__main__':
	try:
		stats = habistat()
	except Exception as e:
		stats = 'Unable to query HABIS status ' + str(e)
	print(stats)
