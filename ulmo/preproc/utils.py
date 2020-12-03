""" Utilities for pre-processing steps"""

import numpy as np

from skimage.restoration import inpaint as sk_inpaint
from scipy.ndimage import median_filter
from scipy import special
from skimage.transform import downscale_local_mean
from skimage import filters

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
        mask;  True = bad

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
                  expon=None, only_inpaint=False, gradient=False,
                  log_scale=False, **kwargs):
    """
    Preprocess an input field image with a series of steps:
        1. Inpainting
        2. Median
        3. Downscale
        4. Sigmoid
        5. Scale
        6. Remove mean
        7. Sobel
        8. Log

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
    scale : float
        Scale the SSTa values by this multiplicative factor
    expon : float
        Exponate the SSTa values by this exponent
    gradient : bool, optional
        If True, apply a Sobel gradient enhancing filter
    **kwargs : catches extraction keywords

    Returns
    -------
    pp_field, meta_dict : np.ndarray, dict
        Pre-processed field, mean temperature

    """
    meta_dict = {}
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

    # Capture more metadata
    srt = np.argsort(field.flatten())
    meta_dict['Tmax'] = field.flatten()[srt[-1]]
    meta_dict['Tmin'] = field.flatten()[srt[0]]
    i10 = int(0.1*field.size)
    i90 = int(0.9*field.size)
    meta_dict['T10'] = field.flatten()[srt[i10]]
    meta_dict['T90'] = field.flatten()[srt[i90]]

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
    meta_dict['mu'] = mu

    # Sigmoid?
    if sigmoid:
        pp_field = special.erf(pp_field)

    # Scale?
    if scale is not None:
        pp_field *= scale

    # Exponate?
    if expon is not None:
        neg = pp_field < 0.
        pos = np.logical_not(neg)
        pp_field[pos] = pp_field[pos]**expon
        pp_field[neg] = -1 * (-1*pp_field[neg])**expon

    # Sobel Gradient?
    if gradient:
        pp_field = filters.sobel(pp_field)
        # Meta
        srt = np.argsort(pp_field.flatten())
        i10 = int(0.1*pp_field.size)
        i90 = int(0.9*pp_field.size)
        meta_dict['G10'] = pp_field.flatten()[srt[i10]]
        meta_dict['G90'] = pp_field.flatten()[srt[i90]]
        meta_dict['Gmax'] = pp_field.flatten()[srt[-1]]

    # Log?
    if log_scale:
        if not gradient:
            raise IOError("Only implemented with gradient=True so far")
        # Set 0 values to the lowest non-zero value
        zero = pp_field == 0.
        if np.any(zero):
            min_nonz = np.min(pp_field[np.logical_not(zero)])
            pp_field[zero] = min_nonz
        # Take log
        pp_field = np.log(pp_field)


    # Return
    return pp_field, meta_dict

