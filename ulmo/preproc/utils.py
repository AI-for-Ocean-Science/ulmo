""" Utilities for pre-processing steps"""

import numpy as np
import os

import pandas
import h5py

from skimage.restoration import inpaint as sk_inpaint
from scipy.ndimage import median_filter
from scipy import special
from skimage.transform import downscale_local_mean
from skimage import filters
from sklearn.utils import shuffle

from ulmo import defs as ulmo_defs

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
    qual_thresh : int, optional
        Quality threshold value;  qual must exceed this
    temp_bounds : tuple
        Temperature interval considered valid

    Returns
    -------
    masks : np.ndarray
        mask;  True = bad

    """
    # Deal with NANs
    sst[np.isnan(sst)] = np.nan
    if qual is not None:
        #qual[np.isnan(qual)] = np.nan
        masks = np.logical_or(np.isnan(sst), np.isnan(qual))
    else:
        masks = np.isnan(sst)

    # Temperature bounds and quality
    qual_masks = np.zeros_like(masks)
    if qual is not None and qual_thresh is not None:
        qual_masks[~masks] = (qual[~masks] > qual_thresh) | (sst[~masks] <= temp_bounds[0]) | (sst[~masks] > temp_bounds[1])
    else:
        qual_masks[~masks] = (sst[~masks] <= temp_bounds[0]) | (sst[~masks] > temp_bounds[1])

    # Finish
    masks = np.logical_or(masks, qual_masks)

    # Return
    return masks

def prep_table_for_preproc(tbl, preproc_root, field_size=None):
    # Prep Table
    for key in ['filename', 'pp_file']:
        if key not in tbl.keys():
            tbl[key] = ''
    tbl['pp_root'] = preproc_root
    if field_size is not None:
        tbl['field_size'] = field_size[0]
    tbl['pp_idx'] = -1
    tbl['pp_type'] = ulmo_defs.mtbl_dmodel['pp_type']['init']
    # 
    return tbl

def preproc_image(item:tuple, pdict:dict, use_mask=False):
    """
    Simple wrapper for preproc_field()
    Mainly for multi-processing

    Parameters
    ----------
    item : tuple
        field, idx or field,mask,idx (use_mask=True)
    pdict : dict
        Preprocessing dict
    use_mask : bool, optional
        If True, allow for an input mask

    Returns
    -------
    pp_field, idx, meta : np.ndarray, int, dict

    """
    # Unpack
    if use_mask:
        field, mask, idx = item
    else:
        field, idx = item
        mask = None

    # Run
    pp_field, meta = preproc_field(field, mask, **pdict)

    # Failed?
    if pp_field is None:
        return None

    # Return
    return pp_field.astype(np.float32), idx, meta


def preproc_field(field, mask, inpaint=True, median=True, med_size=(3,1),
                  downscale=True, dscale_size=(2,2), sigmoid=False, scale=None,
                  expon=None, only_inpaint=False, gradient=False,
                  noise=None,
                  log_scale=False, **kwargs):
    """
    Preprocess an input field image with a series of steps:
        1. Inpainting
        2. Add noise
        3. Median
        4. Downscale
        5. Sigmoid
        6. Scale
        7. Remove mean
        8. Sobel
        9. Log

    Parameters
    ----------
    field : np.ndarray
    mask : np.ndarray or None
        Data mask.  True = masked
        Required for inpainting
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
    noise : float, optional
        If provided, add white noise with this value
    scale : float, optional
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

    # Add noise?
    if noise is not None:
        field += np.random.normal(loc=0., 
                                  scale=noise, 
                                  size=field.shape)
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


def write_pp_fields(pp_fields:list, meta:list, 
                    main_tbl:pandas.DataFrame, 
                    ex_idx:np.ndarray,
                    ppf_idx:np.ndarray,
                    valid_fraction:float,
                    s3_file:str, local_file:str):
    
    # Recast
    pp_fields = np.stack(pp_fields)
    pp_fields = pp_fields[:, None, :, :]  # Shaped for training

    print("After pre-processing, there are {} images ready for analysis".format(pp_fields.shape[0]))
    
    # Fill up
    main_tbl.loc[ex_idx, 'pp_file'] = s3_file

    # Ordered index by current order of pp_fields
    idx_idx = ex_idx[ppf_idx]

    # Mu
    main_tbl['mean_temperature'] = [imeta['mu'] for imeta in meta]
    clms = list(main_tbl.keys())
    # Others
    for key in ['Tmin', 'Tmax', 'T90', 'T10']:
        if key in meta[0].keys():
            main_tbl.loc[idx_idx, key] = [imeta[key] for imeta in meta]
            # Add to clms
            if key not in clms:
                clms += [key]

    # Train/validation
    n = int(valid_fraction * pp_fields.shape[0])
    idx = shuffle(np.arange(pp_fields.shape[0]))
    valid_idx, train_idx = idx[:n], idx[n:]

    # Update table
    main_tbl.loc[idx_idx[valid_idx], 'pp_idx'] = np.arange(valid_idx.size)
    main_tbl.loc[idx_idx[train_idx], 'pp_idx'] = np.arange(train_idx.size)
    main_tbl.loc[idx_idx[valid_idx], 'pp_type'] = ulmo_defs.mtbl_dmodel['pp_type']['valid']
    main_tbl.loc[idx_idx[train_idx], 'pp_type'] = ulmo_defs.mtbl_dmodel['pp_type']['train']

    # ###################
    # Write to disk (avoids holding another 20Gb in memory)
    with h5py.File(local_file, 'w') as f:
        # Validation
        f.create_dataset('valid', data=pp_fields[valid_idx].astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', data=main_tbl.iloc[valid_idx].to_numpy(dtype=str).astype('S'))
        dset.attrs['columns'] = clms
        # Train
        if valid_fraction < 1:
            f.create_dataset('train', data=pp_fields[train_idx].astype(np.float32))
            dset = f.create_dataset('train_metadata', data=main_tbl.iloc[train_idx].to_numpy(dtype=str).astype('S'))
            dset.attrs['columns'] = clms
    print("Wrote: {}".format(local_file))

    # Return
    return main_tbl
