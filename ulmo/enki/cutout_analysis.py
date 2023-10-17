""" Analysis of individual cutout images 
See analysis.py for higher level routines
"""
import os
import numpy as np
import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from scipy.sparse import csc_matrix
from scipy.interpolate import griddata
from skimage.restoration import inpaint as sk_inpaint

from ulmo.enki import enki_utils

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
        keys (list, optional):
            Keys to use for each file. Defaults to None which means ['valid']*3

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
               bias_value:float=0., keys:list=None):
    """ Calculate the RMS of the cutouts
               bias_value:float=0., keys:list=None):

    Args:
        f_orig (h5py.File): Pointer to original images
        f_recon (h5py.File): Pointer to reconstructed images
        f_mask (h5py.File): Pointer to mask images
        patch_sz (int, optional): patch size. Defaults to 4.
        bias_value (float, optional): Value to subtract from the difference. Defaults to 0.
            Defined as recon-orig (see measure_bias below)
        keys (list, optional): Keys to use for each file. Defaults to ['valid']*3.
        

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


def rms_single_img(orig_img:np.array, recon_img:np.array, mask_img:np.array, patch_sz=4):
    """ Calculate the RMS of a single image

    Args:
        orig_img (np.array): Original image
        recon_img (np.array): Reconstructed image
        mask_img (np.array): Mask image
        patch_sz (int, optional): Patch size. Defaults to 4.

    Returns:
        float: RMS value
    """
    # remove edges
    orig_img  = orig_img[patch_sz:-patch_sz, patch_sz:-patch_sz]
    recon_img = recon_img[patch_sz:-patch_sz, patch_sz:-patch_sz]
    mask_img  = mask_img[patch_sz:-patch_sz, patch_sz:-patch_sz]
    
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

def simple_inpaint(items:list, 
                   inpaint_type:str='biharmonic'):
    """ Simple inpainting usually used with multi-processing

    Args:
        items (list): 
            List of [img, mask]
            Built this way for multi-processing
        inpaint_type (str, optional): 
            Inpainting type.  Defaults to 'biharmonic'.
            biharmonic : sk_inpaint.inpaint_biharmonic

    Returns:
        np.ndarray: Inpainted image
    """
    # Unpack
    img, mask = items

    # Do it
    if inpaint_type == 'biharmonic':
        return sk_inpaint.inpaint_biharmonic(
            img, mask, channel_axis=None)
    elif inpaint_type[0:4] == 'grid':
        flavor = inpaint_type.split('_')[1]
        unmasked = np.where(mask == 0)
        x_pts = unmasked[0]
        y_pts = unmasked[1]
        vals = img[unmasked] 
        # Interpolate
        all_x, all_y = np.meshgrid(np.arange(img.shape[0]), 
                                   np.arange(img.shape[1]), 
                                   indexing='ij')
        # Do it
        return griddata((x_pts, y_pts), vals, (all_x, all_y), method=flavor) 
    else:
        raise IOError("Bad inpainting type")


def inpaint_images(inpaint_file:str, 
            t:int, p:int, dataset:str,
            debug:bool=False, 
            method:str='biharmonic',
            patch_sz:int=4, n_cores:int=10, 
            nsub_files:int=5000):
    """ Inpaint images with one of the inpainting methods

    Args:
        inpaint_file (str):  Output file
        t (int): training percentile
        p (int): mask percentile
        dataset (str): dataset ['VIIRS', 'LLC', 'LLC2_nonoise]
        method (str, optional): Inpainting method. Defaults to 'biharmonic'.
        debug (bool, optional): Debug?. Defaults to False.
        patch_sz (int, optional): patch size. Defaults to 4.
        n_cores (int, optional): number of cores. Defaults to 10.
        nsub_files (int, optional): Number of sub files Defaults to 5000.
    """


    # Load images
    tbl_file, orig_file, recon_file, mask_file = enki_utils.set_files(
        dataset, t, p)
    print(f"Inpainting {orig_file} with {method}")

    f_orig = h5py.File(orig_file, 'r')
    #f_recon = h5py.File(recon_file,'r')
    f_mask = h5py.File(mask_file,'r')

    if debug:
        nfiles = 1000
        nsub_files = 100
        orig_imgs = f_orig['valid'][:nfiles,0,...]
        #recon_imgs = f_recon['valid'][:nfiles,0,...]
        mask_imgs = f_mask['valid'][:nfiles,0,...]
    else:
        print("Loading images...")
        orig_imgs = f_orig['valid'][:,0,...]
        #recon_imgs = f_recon['valid'][:,0,...]
        mask_imgs = f_mask['valid'][:,0,...]

    nfiles = orig_imgs.shape[0]

    # Analyze
    #diff_recon = (orig_imgs - recon_imgs)*mask_imgs

    # Inpatinting
    map_fn = partial(simple_inpaint, inpaint_type=method)

    nloop = nfiles // nsub_files + ((nfiles % nsub_files) > 0)
    inpainted = []
    for kk in range(nloop):
        i0 = kk*nsub_files
        i0 = kk*nsub_files
        i1 = min((kk+1)*nsub_files, nfiles)
        print('Files: {}:{} of {}'.format(i0, i1, nfiles))
        sub_files = [(orig_imgs[ii,...], mask_imgs[ii,...]) for ii in range(i0, i1)]
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(
                sub_files) // n_cores if len(sub_files) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, sub_files,
                                                chunksize=chunksize), total=len(sub_files)))
        # Save
        inpainted.append(np.array(answers))
    # Collate
    inpainted = np.concatenate(inpainted)

    # Save
    if not debug:
        with h5py.File(inpaint_file, 'w') as f:
            # Validation
            f.create_dataset('inpainted', data=inpainted.astype(np.float32))
        print(f'Wrote: {inpaint_file}')
    else:
        embed(header='297 of cutout_analysis.py')
 