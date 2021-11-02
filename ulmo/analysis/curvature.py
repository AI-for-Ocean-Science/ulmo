# curvature, overlap of convex hulls, and intersect

from scipy.interpolate import UnivariateSpline
import numpy as np
import cvxpy as cvxpy

def curvature_splines(x, y=None, error=0.1):
    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.
    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    error : float
        The admisible error when interpolating the splines
    Returns
    -------
    curvature: numpy.array shape (n_points, )
    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """

    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=3, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=3, w=1 / np.sqrt(std))

    xˈ = fx.derivative(1)(t)
    xˈˈ= fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ= fy.derivative(2)(t)
    curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 3 / 2)
    return curvature


# Determine feasibility of Ax <= b
# cloud1 and cloud2 should be numpy.ndarrays
def clouds_overlap(cloud1, cloud2):
    """
    Determines whether two shapes overlap in space
    
    Parameters
    ----------
    cloud1, cloud2 : (np.ndarrays)
    
    
    Returns
    -------
    True or False
    """
    
    
    # build the A matrix
    cloud12 = np.vstack((-cloud1, cloud2))
    vec_ones = np.r_[np.ones((len(cloud1),1)), -np.ones((len(cloud2),1))]
    A = np.r_['1', cloud12, vec_ones]

    # make b vector
    ntot = len(cloud1) + len(cloud2)
    b = -np.ones(ntot)

    # define the x variable and the equation to be solved
    x = cvxpy.Variable(A.shape[1])
    constraints = [A@x <= b]

    # since we're only determining feasibility there is no minimization
    # so just set the objective function to a constant
    obj = cvxpy.Minimize(0)

    # SCS was the most accurate/robust of the non-commercial solvers
    # for my application
    problem = cvxpy.Problem(obj, constraints)
    problem.solve(solver=cvxpy.SCS)

    # Any 'inaccurate' status indicates ambiguity, so you can
    # return True or False as you please
    if problem.status == 'infeasible' or problem.status.endswith('inaccurate'):
        return True
    else:
        return False


# quick way to check if two triangles intersect
    
class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y

def ccw(A,B,C):
	return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


a = Point(0,0)
b = Point(0,1)
c = Point(1,1)
d = Point(1,0)


print intersect(a,b,c,d)
print intersect(a,c,b,d)
print intersect(a,d,b,c)