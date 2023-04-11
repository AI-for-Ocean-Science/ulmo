""" Analysis of cutout images"""

import numpy as np
import h5py

from IPython import embed

def rms_images(f_orig:h5py.File, f_recon:h5py.File, f_mask:h5py.File, 
               patch_sz:int=4):
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
    orig_imgs = f_orig['valid'][:,0,...]
    recon_imgs = f_recon['valid'][:,0,...]
    mask_imgs = f_mask['valid'][:,0,...]

    # Mask out edges
    mask_imgs[:, patch_sz:-patch_sz,patch_sz:-patch_sz] = 0

    # Analyze
    print("Calculate")
    calc = (orig_imgs - recon_imgs)*mask_imgs

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