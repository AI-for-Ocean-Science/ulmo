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

def cutout_vel_stat(item:tuple):
    """
    Simple function to measure velocity stats
    Enable multi-processing

    Parameters
    ----------
    item : tuple
        U_cutout, V_cutout, idx

    Returns
    -------
    idx, stats : int, dict

    """
    # Unpack
    U_cutout, V_cutout, idx = item

    # Deal with nan
    gdU = np.isfinite(U_cutout)
    gdV = np.isfinite(V_cutout)

    # Stat dict
    v_stats = {}
    v_stats['U_mean'] = np.mean(U_cutout[gdU])
    v_stats['V_mean'] = np.mean(V_cutout[gdV])
    v_stats['U_rms'] = np.std(U_cutout[gdU])
    v_stats['V_rms'] = np.std(V_cutout[gdV])
    UV_cutout = np.sqrt(U_cutout**1 + U_cutout**2)
    v_stats['UV_mean'] = np.mean(UV_cutout[gdU & gdV])
    v_stats['UV_rms'] = np.std(UV_cutout[gdU & gdV])

    # Return
    return idx, v_stats

def cutout_curl(item:tuple, pdict=None):
    """
    Simple function to measure velocity stats
    Enable multi-processing

    Parameters
    ----------
    item : tuple
        U_cutout, V_cutout, idx

    Returns
    -------
    idx, stats : int, dict

    """
    # Unpack
    U_cutout, V_cutout, idx = item

    # Deal with nan
    badU = np.isnan(U_cutout)
    badV = np.isnan(V_cutout)
    U_cutout[badU] = 0.
    V_cutout[badV] = 0.

    curl = calc_curl(U_cutout, V_cutout)
    return idx, curl