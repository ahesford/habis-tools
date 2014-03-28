'''
Routines for performing acoustic trilateration.
'''

import numpy as np, math
from numpy import fft, linalg as la
from scipy.sparse.linalg import lsmr
from pyajh import cutil

def jacobian(centers, pos, times=None, c=None):
	'''
	Computes the Jacobian matrix for Newton-Raphson iteration to
	acoustically trilaterate the position of one element relative to any
	number of known elements with centers in the rank-2 array (or list of
	arrays or lists)

		centers = [c1, c2, ..., cn].

	If times is provided, it specifies a list of round-trip arrival times
	[t1, t2, ..., tn] used to characterize the separation between each
	center and the element to be located. A sound speed c that relates the
	arrival times to distances must be specified when times is specified.
	The resulting Jacobian will contain one column for each position
	variable, plus one extra column that accounts for a variable delay
	associated with the element to be located.

	The position estimate is a rank-1 array or list and takes the form 

		pos = [x1, ..., xj, tau],

	where x1, ..., xj are estimates of position variables and tau is an
	estimate of the delay. If times is None, the delay tau is optional and
	may be omitted from pos; it will be ignored if present.

	The Jacobian takes the form

		J = 2 * [ D[i,j] | T[i] ],

	where T[i] is a column vector with elements

		Ti = c**2 * (tau - times[i] / 2)

	and is only included when times is not None. The remaining entries

		D[i,j] = centers[i][j] - pos[j].
	'''
	if times is not None:
		# Ensure that c is provided whenever arrival times are provided
		if c is None:
			raise TypeError('Speed c must be provided with times')
		# Ensure the times list is a row vector
		try: times = times.ravel()
		except AttributeError: times = np.array(times).ravel()

	# Ensure that pos is a row vector
	try: pos = pos.ravel()
	except AttributeError: pos = np.array(pos).ravel()

	# Ensure that the centers argument is a rank-2 array
	try: rank = len(centers.shape)
	except AttributeError:
		centers = np.array(centers)
		rank = len(centers.shape)
	if rank != 2: raise TypeError('Element centers must be a rank-2 array')

	# The Jacobian has an extra column if tau varies
	nrows, ndim = centers.shape
	ncols = ndim + (1 if times is not None else 0)
	if len(pos) < ncols:
		raise TypeError('Dimensionality of pos too small for Jacobian size')
	jac = np.empty((nrows, ncols), dtype=pos.dtype)
	# Build the positional contributions to the Jacobian
	jac[:,:ndim] = 2 * (centers - pos[np.newaxis,:ndim])

	if times is not None:
		# Build the delay contribution to the Jacobian
		jac[:,ndim] = 2 * c**2 * (pos[ndim] - times / 2.)

	return jac


def costfunc(centers, pos, times, c):
	'''
	Computes the cost function associated with Newton-Raphson iterations
	for acoustic trilateration. The rank-2 array (or list of arrays or
	lists)

		centers = [c1, c2, ..., cn]

	specifies the spatial coordinates of any number of known elements. The
	estimate of the element to be located takes the form

		pos = [x1, ..., xj, tau],

	where x1, ..., xj are estimates of position and tau is an estimate of
	the intrinsic delay associated with measurement. Round-trip arrival
	times are specified in the rank-1 array or list

		times = [t1, t2, ..., tn]

	and are related to separation of elements by a sound speed c.

	The cost function takes the form

		F[i] = c**2 * (times[i] / 2 - tau)**2 
		       - sum((centers[i] - pos[:-1])**2).
	'''
	# Ensure the arguments are properly formatted
	try: times = times.ravel()
	except AttributeError: times = np.array(times).ravel()

	try: pos = pos.ravel()
	except AttributeError: pos = np.array(pos).ravel()

	try: rank = len(centers.shape)
	except AttributeError:
		centers = np.array(centers)
		rank = len(centers.shape)
	if rank != 2: raise TypeError('Element centers must be a rank-2 array')

	nrows, ndim = centers.shape
	tau = 0.
	if len(pos) > ndim: tau = pos[ndim]
	elif len(pos) < ndim:
		raise TypeError('Dimensionality of pos must at least match that of centers')

	return ((c * (times / 2. - tau))**2
			- np.sum((centers - pos[np.newaxis,:ndim])**2, axis=1))


def newton(centers, times, c, pos=None, maxit=100, tol=1e-6, **kwargs):
	'''
	For known elements with coordinates specified in the rank-2 array

		centers = [c1, c2, ..., cn],

	use Newton-Raphson iteration to recover the position of an unknown
	element associated with round-trip arrival times

		times = [t1, t2, ..., tn]

	through a medium with sound speed c.

	An initial estimate pos may be specified, but is assumed to be the
	origin by default. Iterations will stop after maxit iterations or when
	the norm of a computed update is less than tol times the norm of the
	guess used to produce the update, whichever occurs first.

	The inverse of the Jacobian is computed using the LSMR algorithm
	(scipy.sparse.linalg.lsmr). Additional kwargs are passed directly to
	the LSMR function.
	'''
	try: rank = len(centers.shape)
	except AttributeError:
		centers = np.array(centers)
		rank = len(centers.shape)
	if rank != 2: raise TypeError('Element centers must be a rank-2 array')

	# Ensure that a copy is made if a position guess was specified
	if pos is not None: pos = np.array(pos)
	else: pos = np.zeros((centers.shape[-1],), dtype=centers.dtype)

	for i in range(maxit):
		# Build the Jacobian and right-hand side
		jac = jacobian(centers, pos)
		cost = costfunc(centers, pos, times, c)
		# Use LSMR to invert the system
		delt = lsmr(jac, cost, **kwargs)[0]
		# Check for convergence
		conv = (la.norm(delt) < tol * la.norm(pos))
		pos -= delt
		if conv: break

	return pos