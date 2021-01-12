""" Methods for extracting fields from full images"""

import numpy as np
from scipy.ndimage import uniform_filter

from IPython import embed


def clear_grid(mask, field_size, method, CC_max=0.05,
                 nsgrid_draw=1, return_fracCC=False):
    """

    Parameters
    ----------
    mask : np.ndarray
    field_size : int
    method : str
        'random'
        'lower_corner'
    CC_max : float
        Maximum cloudy fraction allowed
    ndraw_mnx
    nsgrid_draw : int, optional
        Number of fields to draw per sub-grid
    return_fracCC : bool, optional
        Return the fraction of the image satisfying the CC value

    Returns
    -------

    """
    # Some checks
    if nsgrid_draw > 1 and method == 'lower_corner':
        raise IOError("Not ready for this..")

    # Sum across the image
    CC_mask = uniform_filter(mask.astype(float), field_size, mode='constant', cval=1.)

    # Clear
    mask_edge = np.zeros_like(mask)
    mask_edge[:field_size//2,:] = True
    mask_edge[-field_size//2:,:] = True
    mask_edge[:,-field_size//2:] = True
    mask_edge[:,:field_size//2] = True
    clear = (CC_mask <= CC_max) & np.invert(mask_edge)  # Added the = sign on 2021-01-12
    if return_fracCC:
        return np.sum(clear)/((clear.shape[0]-field_size)*(clear.shape[1]-field_size))

    # Indices
    idx_clear = np.where(clear)
    nclear = idx_clear[0].size
    keep = np.zeros_like(idx_clear[0], dtype=bool)

    # Enough clear?
    if nclear < nsgrid_draw:
        return None, None, None

    # Sub-grid me
    sub_size = field_size // 4
    rows = np.arange(mask.shape[0]) // sub_size + 1
    sub_nrows = rows[-1]  # The 1 was already added in
    cols = np.arange(mask.shape[1]) // sub_size * rows[-1]
    sub_grid = np.outer(rows, np.ones(mask.shape[1], dtype=int)) + np.outer(
        np.ones(mask.shape[0], dtype=int), cols)

    # Work through each sub_grid
    sub_values = sub_grid[idx_clear]
    uni_sub, counts = np.unique(sub_values, return_counts=True)
    for iuni, icount in zip(uni_sub, counts):
        mt = np.where(sub_values == iuni)[0]
        if method == 'random':
            r_idx = np.random.choice(icount, size=min(icount, nsgrid_draw))
            keep[mt[r_idx]] = True
        elif method == 'center':
            # Grid lower corners
            sgrid_col = (iuni - 1) // sub_nrows
            sgrid_row = (iuni-1) - sgrid_col*sub_nrows
            # i,j center
            iirow = sgrid_row * sub_size + sub_size // 2
            jjcol = sgrid_col * sub_size + sub_size // 2
            # Distanc3
            dist2 = (idx_clear[0][mt]-iirow)**2 + (idx_clear[1][mt]--jjcol)**2
            # Min and keep
            imin = np.argmin(dist2)
            keep[mt[imin]] = True
        else:
            raise IOError("Bad method option")

    # Offset to lower left corner
    picked_row = idx_clear[0][keep] - field_size//2
    picked_col = idx_clear[1][keep] - field_size//2

    # Return
    return picked_row, picked_col, CC_mask[idx_clear][keep]



