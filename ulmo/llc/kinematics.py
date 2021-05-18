""" Routines related to kinemaic measures of LLC data """
import numpy as np

def calc_div(U, V):
    dUdx = np.gradient(U, axis=1)
    dVdy = np.gradient(V, axis=0)
    div = dUdx + dVdy
    #
    return div

def calc_curl(U, V):
    dUdy = np.gradient(U, axis=0)
    dVdx = np.gradient(V, axis=1)
    curl = dVdx - dUdy
    # Return
    return curl

def calc_normal_strain(U, V):
    dUdx = np.gradient(U, axis=1)
    dVdy = np.gradient(V, axis=0)
    norm_strain = dUdx - dVdy
    # Return
    return norm_strain