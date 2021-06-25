""" Routines related to kinemaic measures of LLC data """
import numpy as np

def calc_div(U:np.ndarray, V:np.ndarray):
    """Calculate the divergence

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: Divergence array
    """
    dUdx = np.gradient(U, axis=1)
    dVdy = np.gradient(V, axis=0)
    div = dUdx + dVdy
    #
    return div

def calc_curl(U:np.ndarray, V:np.ndarray):  # Also the relative or vertical vorticity?!
    """Calculate the curl (aka relative vorticity)

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: Curl
    """
    dUdy = np.gradient(U, axis=0)
    dVdx = np.gradient(V, axis=1)
    curl = dVdx - dUdy
    # Return
    return curl

def calc_normal_strain(U:np.ndarray, V:np.ndarray):
    """Calculate the normal strain

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: normal strain
    """
    dUdx = np.gradient(U, axis=1)
    dVdy = np.gradient(V, axis=0)
    norm_strain = dUdx - dVdy
    # Return
    return norm_strain

def calc_shear_strain(U:np.ndarray, V:np.ndarray):
    """Calculate the shear strain

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: shear strain
    """
    dUdy = np.gradient(U, axis=0)
    dVdx = np.gradient(V, axis=1)
    shear_strain = dUdy + dVdx
    # Return
    return shear_strain

def calc_lateral_strain_rate(U:np.ndarray, V:np.ndarray):
    """Calculate the lateral strain rate

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: alpha
    """
    dUdx = np.gradient(U, axis=1)
    dUdy = np.gradient(U, axis=0)
    dVdx = np.gradient(V, axis=1)
    dVdy = np.gradient(V, axis=0)
    #
    alpha = np.sqrt((dUdx-dVdy)**2 + (dVdx+dUdy)**2)
    return alpha

def calc_okubo_weiss(U:np.ndarray, V:np.ndarray):
    """Calculate Okubo-Weiss

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field

    Returns:
        np.ndarray: okubo-weiss
    """
    s_n = calc_normal_strain(U, V)
    s_s = calc_shear_strain(U, V)
    w = calc_curl(U, V)  # aka relative vorticity
    #
    W = s_n**2 + s_s**2 - w**2
    # Return
    return W