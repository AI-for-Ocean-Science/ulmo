""" Analysis of cutout images"""

import numpy as np
import h5py

from IPython import embed

def rms_images(f_orig:h5py.File, f_recon:h5py.File, f_mask:h5py.File, 
               patch_sz:int=4, nimgs:int=None, debug:bool=False):
    """_summary_

    Args:
        f_orig (h5py.File): Pointer to original images
        f_recon (h5py.File): Pointer to reconstructed images
        f_mask (h5py.File): Pointer to mask images
        patch_sz (int, optional): patch size. Defaults to 4.

    Returns:
        np.array: RMS values
    """
    # Load em all
    print("Loading images...")
    if nimgs is None:
        nimgs = f_orig['valid'].shape[0]

    # Grab em
    orig_imgs = f_orig['valid'][:nimgs,0,...]
    recon_imgs = f_recon['valid'][:nimgs,0,...]
    mask_imgs = f_mask['valid'][:nimgs,0,...]

    # Mask out edges
    print("Masking edges")
    mask_imgs[:, 0:patch_sz, :] = 0
    mask_imgs[:, -patch_sz:, :] = 0
    mask_imgs[:, :, 0:patch_sz] = 0
    mask_imgs[:, :, -patch_sz:] = 0

    # Analyze
    print("Calculate")
    calc = (orig_imgs - recon_imgs)*mask_imgs

    #if debug:
    #    embed(header='43 of cutout_analysis.py')

    # Square
    print("Square")
    calc = calc**2

    # Mean
    print("Mean")
    nmask = np.sum(mask_imgs, axis=(1,2))
    calc = np.sum(calc, axis=(1,2)) / nmask

    # RMS
    print("Root")
    return np.sqrt(calc)