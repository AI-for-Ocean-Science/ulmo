""" Analysis of cutout images"""

import numpy as np
import h5py
from scipy.sparse import csc_matrix

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


def rms_single_img(orig_img, recon_img, mask_img):
    """ Calculate rms of a single image (ignore edges)
        orig_img:  original img (64x64)
        recon_img: reconstructed image (64x64)
        mask_img:  mask of recon_image (64x64)
    """
    # remove edges
    orig_img  = orig_img[4:-4, 4:-4]
    recon_img = recon_img[4:-4, 4:-4]
    mask_img  = mask_img[4:-4, 4:-4]
    
    # Find i,j positions from mask
    mask_sparse = csc_matrix(mask_img)
    mask_i,mask_j = mask_sparse.nonzero()
    
    # Find differences
    diff = np.zeros(len(mask_i))
    for idx, (i, j) in enumerate(zip(mask_i, mask_j)):
        diff[idx] = orig_img[i,j] - recon_img[i,j]
    
    #embed(header='44 of anly_rms.py')
    diff = np.square(diff)
    rms = diff.mean()
    rms = np.sqrt(rms)
    return rms