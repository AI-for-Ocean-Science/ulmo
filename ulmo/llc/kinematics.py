""" Routines related to kinemaic measures of LLC data """
import numpy as np

def calc_div(U, V):
    dUdx = np.gradient(U, axis=1)
    dVdy = np.gradient(V, axis=0)
    div = dUdx + dVdy
    #
    return div

def calc_curl(U, V):  # Also the relative or vertical vorticity?!
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

def calc_shear_strain(U, V):
    dUdy = np.gradient(U, axis=0)
    dVdx = np.gradient(V, axis=1)
    shear_strain = dUdy + dVdx
    # Return
    return shear_strain

def calc_lateral_strain_rate(U, V):
    dUdx = np.gradient(U, axis=1)
    dUdy = np.gradient(U, axis=0)
    dVdx = np.gradient(V, axis=1)
    dVdy = np.gradient(V, axis=0)
    #
    alpha = np.sqrt((dUdx-dVdy)**2 + (dVdx+dUdy)**2)
    return alpha

def calc_okubo_weiss(U, V):
    s_n = calc_normal_strain(U, V)
    s_s = calc_shear_strain(U, V)
    w = calc_curl(U, V)  # aka relative vorticity
    #
    W = s_n**2 + s_s**2 - w**2
    # Return
    return W