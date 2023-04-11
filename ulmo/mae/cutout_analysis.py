""" Analysis of cutout images"""

import numpy as np

from IPython import embed

def rms_images(f_orig, f_recon, f_mask, patch_sz:int=4):
    # Load em all
    orig_imgs = f_orig['valid'][:,0,...]
    recon_imgs = f_recon['valid'][:,0,...]
    mask_imgs = f_mask['valid'][:,0,...]

    # Mask out edges
    mask_imgs[:, patch_sz:-patch_sz,patch_sz:-patch_sz] = 0

    # Analyze
    calc = (orig_imgs - recon_imgs)*mask_imgs

    embed(header='19 of cutout_analysis.py')

    # Square
    calc = calc**2

    # Mean
    nmask = np.sum(mask_imgs, axis=(1,2))
    calc = np.sum(calc, axis=(1,2)) / nmask

    # RMS
    return np.sqrt(calc)