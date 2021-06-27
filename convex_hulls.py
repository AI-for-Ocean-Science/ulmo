import numpy as np
import cvxpy as cvxpy

# Determine feasibility of Ax <= b
# cloud1 and cloud2 should be numpy.ndarrays
def clouds_overlap(cloud1, cloud2):
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