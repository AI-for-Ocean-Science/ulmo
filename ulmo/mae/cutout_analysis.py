""" Analysis of cutout images"""

import numpy as np
import h5py
from scipy.sparse import csc_matrix

from IPython import embed

def load_cutouts(f_orig:h5py.File, f_recon:h5py.File, f_mask:h5py.File, 
               nimgs:int=None, debug:bool=False, patch_sz:int=None,
               keys:list=None):
    """ Load the cutouts

    Args:
        f_orig (h5py.File): 
            Pointer to original images
        f_recon (h5py.File): 
            Pointer to reconstructed images
        f_mask (h5py.File): 
            Pointer to mask images
        nimgs (int, optional): 
            Number of images to load. Defaults to None which means all
        debug (bool, optional): 
            Debugging flag. Defaults to False.
        patch_sz (int, optional): 
            patch size. Defaults to None.

    Returns:
        tuple: orig_imgs, recon_imgs, mask_imgs
    """
    if keys is None:
        keys = ['valid']*3
    # Load em all
    print("Loading images...")
    if nimgs is None:
        nimgs = f_orig[keys[0]].shape[0]

    # Grab em
    orig_imgs = f_orig[keys[0]][:nimgs,0,...]
    if len(f_recon[keys[1]].shape) == 4:
        recon_imgs = f_recon[keys[1]][:nimgs,0,...]
    else:
        recon_imgs = f_recon[keys[1]][:nimgs,...]
    mask_imgs = f_mask[keys[2]][:nimgs,0,...]

    # Mask out edges
    print("Masking edges")
    if patch_sz is not None:
        mask_imgs[:, 0:patch_sz, :] = 0
        mask_imgs[:, -patch_sz:, :] = 0
        mask_imgs[:, :, 0:patch_sz] = 0
        mask_imgs[:, :, -patch_sz:] = 0

    return orig_imgs, recon_imgs, mask_imgs


def rms_images(f_orig:h5py.File, f_recon:h5py.File, f_mask:h5py.File, 
               patch_sz:int=4, nimgs:int=None, debug:bool=False, 
               bias_value:float=0.):
    """ Calculate the RMS of the cutouts
               bias_value:float=0., keys:list=None):

    Args:
        f_orig (h5py.File): Pointer to original images
        f_recon (h5py.File): Pointer to reconstructed images
        f_mask (h5py.File): Pointer to mask images
        patch_sz (int, optional): patch size. Defaults to 4.
        bias_value (float, optional): Value to subtract from the difference. Defaults to 0.
            Defined as recon-orig (see measure_bias below)

    Returns:
        np.array: RMS values
    """
    # Load (and mask) images
    orig_imgs, recon_imgs, mask_imgs = load_cutouts(
        f_orig, f_recon, f_mask, nimgs=nimgs, patch_sz=patch_sz, keys=keys)

    # Analyze
    print("Calculate")
    calc = (recon_imgs - orig_imgs)*mask_imgs - bias_value

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


def measure_bias(f_orig, f_recon, f_mask, patch_sz=4,
                 nimgs:int=None, debug:bool=False):
    """ Measure the bias in cutouts
    WARNING:  THIS IS REPEATED IN bias.py

    Args:
        f_orig (_type_): 
            Pointer to original images
        f_recon (_type_): 
            Pointer to reconstructed images
        f_mask (_type_): _description_
        patch_sz (int, optional): _description_. Defaults to 4.

    Returns:
        tuple: median_bias, mean_bias
    """
    # Load (and mask) images
    orig_imgs, recon_imgs, mask_imgs = load_cutouts(
        f_orig, f_recon, f_mask, nimgs=nimgs, patch_sz=patch_sz)

    # Difference
    diff_true = (recon_imgs - orig_imgs)*mask_imgs

    # Stats
    #if debug:
    #    embed(header='120 of cutout_analysis.py')
    median_bias = np.median(diff_true[np.abs(diff_true) > 0.])
    mean_bias = np.mean(diff_true[np.abs(diff_true) > 0.])

    #mean_img = np.mean(orig_img[np.isclose(mask_img,0.)])
    #stats['mean_img'] = mean_img
    # Return
    return median_bias, mean_bias