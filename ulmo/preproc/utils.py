""" Utilities for pre-processing steps"""

import numpy as np

from skimage.restoration import inpaint as sk_inpaint
from scipy.ndimage import median_filter
from skimage.transform import downscale_local_mean

def build_mask(sst, qual, qual_thresh=2, temp_bounds=(-2,33)):
    sst[np.isnan(sst)] = np.nan
    qual[np.isnan(qual)] = np.nan
    # Deal with NaN
    masks = np.logical_or(np.isnan(sst), np.isnan(qual))
    # Temperature bounds and quality
    qual_masks = np.zeros_like(masks)
    qual_masks[~masks] = (qual[~masks] > qual_thresh) | (sst[~masks] <= temp_bounds[0]) | (sst[~masks] > temp_bounds[1])
    masks = np.logical_or(masks, qual_masks)
    # Return
    return masks

def preproc_field(field, mask, inpaint=True, median=True, med_size=(3,1),
                  downscale=True, dscale_size=(2,2)):
    # Inpaint?
    if inpaint:
        mask = np.uint8(mask)
        field = sk_inpaint.inpaint_biharmonic(field, mask, multichannel=False)

    # Median
    if median:
        field = median_filter(field, size=med_size)

    # Reduce to 64x64
    if downscale:
        field = downscale_local_mean(field, dscale_size)

    # Check for junk
    if np.isnan(field).sum() > 0:
        return None, None

    # De-mean the field
    mu = np.mean(field)
    field = field - mu

    # Return
    return field, mu


