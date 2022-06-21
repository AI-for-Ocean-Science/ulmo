""" Routines related to kinemaic measures of LLC data """
import numpy as np
from skimage.transform import resize_local_mean

from IPython import embed

try:
    from gsw import density
except ImportError:
    print("gsw not imported;  cannot do density calculations")

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

def calc_gradT(Theta:np.ndarray, dx=2.):
    """Calculate |grad T|^2

    Args:
        Theta (np.ndarray): SST field
        dx (float, optional): Grid spacing in km

    Returns:
        np.ndarray: |grad T|^2 field
    """

    # Gradient
    dTdx = np.gradient(Theta, axis=1) / dx
    dTdy = np.gradient(Theta, axis=0) / dx

    # Magnitude
    grad_T2 = dTdx**2 + dTdy**2
    return grad_T2

def calc_gradb(Theta:np.ndarray, Salt:np.ndarray,
             ref_rho=1025., g=0.0098, dx=2.):
    """Calculate |grad b|^2

    Args:
        Theta (np.ndarray): SST field
        Salt (np.ndarray): Salt field
        ref_rho (float, optional): Reference density
        g (float, optional): Acceleration due to gravity
            in km/s^2
        dx (float, optional): Grid spacing in km

    Returns:
        np.ndarray: |grad b|^2 field
    """
    # Buoyancy
    rho = density.rho(Salt, Theta, np.zeros_like(Salt))
    b = g*rho/ref_rho

    # Gradient
    dbdx = np.gradient(b, axis=1) / dx
    dbdy = np.gradient(b, axis=0) / dx

    # Magnitude
    grad_b2 = dbdx**2 + dbdy**2
    embed(header='136 of kin')
    return grad_b2

def calc_F_s(U:np.ndarray, V:np.ndarray,
             Theta:np.ndarray, Salt:np.ndarray,
             add_gradb=False,
             ref_rho=1025., g=0.0098, dx=2.):
    """Calculate Frontogenesis forcing term

    Args:
        U (np.ndarray): U velocity field
        V (np.ndarray): V velocity field
        SST (np.ndarray): SST field
        Salt (np.ndarray): Salt field
        ref_rho (float, optional): Reference density
        add_gradb (bool, optional): Calculate+return gradb 
        g (float, optional): Acceleration due to gravity
            in km/s^2
        dx (float, optional): Grid spacing in km

    Returns:
        np.ndarray or tuple: F_s field (, gradb2)
    """
    dUdx = np.gradient(U, axis=1)
    dVdx = np.gradient(V, axis=1)
    #
    dUdy = np.gradient(U, axis=0)
    dVdy = np.gradient(V, axis=0)

    # Buoyancy
    rho = density.rho(Salt, Theta, np.zeros_like(Salt))
    dbdx = -1*np.gradient(g*rho/ref_rho, axis=1) / dx
    dbdy = -1*np.gradient(g*rho/ref_rho, axis=0) / dx

    # Terms
    F_s_x = -1 * (dUdx*dbdx + dVdx*dbdy) * dbdx 
    F_s_y = -1 * (dUdy*dbdx + dVdy*dbdy) * dbdy 

    # Finish
    F_s = F_s_x + F_s_y

    # div b too?
    if add_gradb:
        grad_b2 = dbdx**2 + dbdy**2
        return F_s, grad_b2
    else:
        return F_s

def cutout_kin(item:tuple, kin_stats:dict, field_size=None,
               extract_kin=False):
    """Simple function to measure kinematic stats
    So far -- front related stats
    Enables multi-processing

    Args:
        item (tuple): Items for analysis
        kin_stats (dict): kin stats to calculate
        extract_kin (bool, optioal): If True, return
            the extracted cutouts too

    Returns:
        tuple: int, dict if extract_kin is False
            Otherwise, int, dict, np.ndarray, np.ndarray
    """
    # Unpack
    U_cutout, V_cutout, Theta_cutout, Salt_cutout, idx = item

    # F_S
    F_s, gradb = calc_F_s(U_cutout, V_cutout, Theta_cutout, Salt_cutout,
                   add_gradb=True)

    # Resize?
    if field_size is not None:
        F_s = resize_local_mean(F_s, (field_size, field_size))
        gradb = resize_local_mean(gradb, (field_size, field_size))

    # Stats
    kin_metrics = calc_kin_stats(F_s, gradb, kin_stats)

    if extract_kin:
        return idx, kin_metrics, F_s, gradb
    else:
        return idx, kin_metrics


def calc_kin_stats(F_s:np.ndarray, gradb:np.ndarray, stat_dict:dict):
    """Calcualte statistics on the F_s metric

    Args:
        F_s (np.ndarray): F_s cutout
        gradb (np.ndarray): |grad b|^2 cutout
        stat_dict (dict): kin dict of metrics to calculate
        and related parameters

    Returns:
        dict: kin metrics
    """
    kin_metrics = {}

    # Frontogensis
    if 'Fronto_thresh' in stat_dict.keys():
        kin_metrics['FS_Npos'] = int(np.sum(F_s > stat_dict['Fronto_thresh']))
    if 'Fronto_sum' in stat_dict.keys():
        kin_metrics['FS_pos_sum'] = np.sum(F_s[F_s > 0.])

    # Fronts
    if 'Front_thresh' in stat_dict.keys():
        kin_metrics['gradb_Npos'] = int(np.sum(gradb > stat_dict['Front_thresh']))
    #
    return kin_metrics


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
