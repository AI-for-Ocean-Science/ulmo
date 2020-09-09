""" Utilities for pre-processing steps"""

import numpy as np

from skimage.restoration import inpaint as sk_inpaint
from scipy.ndimage import median_filter
from scipy import special
from skimage.transform import downscale_local_mean

from IPython import embed


def build_mask(sst, qual, qual_thresh=2, temp_bounds=(-2,33)):
    """
    Generate a mask based on NaN, qual, and temperature bounds

    Parameters
    ----------
    sst : np.ndarray
        Full SST image
    qual : np.ndarray
        Quality image
    qual_thresh : int
        Quality threshold value;  qual must exceed this
    temp_bounds : tuple
        Temperature interval considered valid


    Returns
    -------
    masks : np.ndarray
        mask

    """
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
                  downscale=True, dscale_size=(2,2), sigmoid=False, scale=None,
                  only_inpaint=False):
    """
    Preprocess an input field image with a series of steps:
        1. Inpainting
        2. Median
        3. Downscale
        4. Sigmoid
        5. Scale
        6. Remove mean

    Parameters
    ----------
    field : np.ndarray
    mask : np.ndarray
        Data mask.  True = masked
    inpaint : bool, optional
        if True, inpaint masked values
    median : bool, optional
        If True, apply a median filter
    med_size : tuple
        Median window to apply
    downscale : bool, optional
        If True downscale the image
    dscale_size : tuple, optional
        Size to rescale by

    Returns
    -------
    pp_field, mu : np.ndarray, float
        Pre-processed field, mean temperature

    """
    # Inpaint?
    if inpaint:
        if mask.dtype.name != 'uint8':
            mask = np.uint8(mask)
        field = sk_inpaint.inpaint_biharmonic(field, mask, multichannel=False)

    if only_inpaint:
        if np.any(np.isnan(field)):
            return None, None
        else:
            return field, None

    # Median
    if median:
        field = median_filter(field, size=med_size)

    # Reduce to 64x64
    if downscale:
        field = downscale_local_mean(field, dscale_size)

    # Check for junk
    if np.any(np.isnan(field)):
        return None, None

    # De-mean the field
    mu = np.mean(field)
    pp_field = field - mu

    # Sigmoid?
    if sigmoid:
        pp_field = special.erf(pp_field)

    # Scale?
    if scale is not None:
        pp_field *= scale

    # Return
    return pp_field, mu


