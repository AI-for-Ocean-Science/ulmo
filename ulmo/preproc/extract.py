""" Methods for extracting fields from full images"""

import numpy as np
from scipy.ndimage import uniform_filter


def random_clear(mask, field_size, CC_max=0.05, ndraw_mnx=(10,1000),
                   min_clear_patch=1000):

    # Sum across the image
    CC_mask = uniform_filter(mask.astype(float), field_size, mode='constant', cval=1.)

    # Clear
    mask_edge = np.zeros_like(mask)
    mask_edge[:field_size//2,:] = True
    mask_edge[-field_size//2:,:] = True
    mask_edge[:,-field_size//2:] = True
    mask_edge[:,:field_size//2] = True
    clear = (CC_mask < CC_max) & np.invert(mask_edge)

    # Indices
    idx_clear = np.where(clear)
    nclear = idx_clear[0].size

    # Enough clear?
    if nclear < min_clear_patch:
        return None, None, None

    ndraw2 = 4 * ((nclear // field_size ** 2) + 1)
    ndraw = int(3000 * nclear / mask.size)
    ndraw = np.maximum(ndraw, ndraw2)
    ndraw = np.minimum(ndraw, ndraw_mnx[1])
    ndraw = np.maximum(ndraw, ndraw_mnx[0])

    # Pick em, randomly
    r_idx = np.random.choice(nclear, size=ndraw)

    # Offset to lower left corner
    picked_row = idx_clear[0][r_idx] - field_size//2
    picked_col = idx_clear[1][r_idx] - field_size//2

    # Return
    return picked_row, picked_col, CC_mask[idx_clear][r_idx]



