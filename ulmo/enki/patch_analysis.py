""" Module for methods related to patch analysis.
Mainly to assess performance of the MAE"""

import numpy as np
import os

import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo import io as ulmo_io

from IPython import embed


def anlayze_full(recon_file,
    orig_file='MAE_LLC_valid_nonoise_preproc.h5',
    stats=['meanT', 'stdT', 'DT', 'median_diff', 
           'std_diff', 'max_diff', 'i_patch', 'j_patch',
           'DT_recon'],
    nsub:int=10000, n_cores:int=4, p_sz:int=4, 
    debug:bool=False, bias:float=0.,
    outfile:str=None):
    """ Analyze the patches in a given file of reconstructed images

    Args:
        recon_file (str): 
            Full s3 path to the reconstruction file
        orig_file (str, optional): _description_. Defaults to 'MAE_LLC_valid_nonoise_preproc.h5'.
        nsub (int, optional): _description_. Defaults to 1000.
        n_cores (int, optional): _description_. Defaults to 4.
        stats (list, optional):
        p_sz (int, optional): _description_. Defaults to 4.
        debug (bool, optional): _description_. Defaults to False.
        outfile (str, optional): Output file. 
            Defaults to None and uses recon_file to generate it.
    """
    mask_file = recon_file.replace('reconstruct', 'mask')

    # Outfile
    if outfile is None:
        outfile = mask_file.replace('mask', 'patches')
        outfile = outfile.replace('.h5', '.npz')

    # Load up
    f_mask = h5py.File(mask_file, 'r')
    f_recon = h5py.File(recon_file, 'r')
    f_orig = h5py.File(orig_file, 'r')
    nimages = f_mask['valid'].shape[0]

    if debug:
        nimages = 10000

    map_fn = partial(patch_stats_img, p_sz=p_sz,
                     stats=stats, bias=bias)

    # Run one to get the number of patches and number of items?
    items = [f_orig['valid'][0,0,...], 
             f_recon['valid'][0,0,...], 
             f_mask['valid'][0,0,...]]
    stat_dict = patch_stats_img(items, p_sz=p_sz, stats=stats)


    # Output array
    npatches = stat_dict[stats[0]].size
    nitems = len(stats)
    output = np.zeros((nimages, npatches, nitems))

    # 
    nloop = nimages // nsub+ ((nimages % nsub) > 0)
    for kk in range(nloop):
        i0 = kk*nsub
        i1 = min((kk+1)*nsub, nimages)
        # Masks
        masks = f_mask['valid'][i0:i1,0,...]
        recons = f_recon['valid'][i0:i1,0,...]
        origs = f_orig['valid'][i0:i1,0,...]
        items = [item for item in zip(origs, recons, masks)]
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
                chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
                answers = list(tqdm(executor.map(map_fn, items,
                                                chunksize=chunksize), total=len(items)))

        # Slurp it in
        for kk, stat_dict in enumerate(answers):
            for nn in range(nitems):
                output[i0+kk,:,nn] = stat_dict[stats[nn]]

    # Write to disk
    np.savez(outfile, data=output, items=stats)
    print(f'Wrote: {outfile}')

    # Upload
    ulmo_io.upload_file_to_s3(
        outfile, 's3://llc/mae/Recon/'+outfile)

# TODO -- Consider using jit on the following method
def find_patches(mask_img, p_sz:int, patch_space:bool=False):
    """ Simple algorithm to find the patches
    in a masked MAE image

    It assumes they are square and whole

    Args:
        mask_img (np.ndarray): Masked image; 1=masked
        p_sz (int): Size of the patch (edge)
        patch_space (bool, optional): Return the patches
        in the patch space.  Defaults to False.
            NOT IMPLEMENTED YET

    Returns:
        list: unRavel'd index of the patches
    """

    flat_mask = mask_img.flatten().astype(int)
    patches = []
    for ss in range(mask_img.size):
        if flat_mask[ss] == 1:
            # Unravel
            i, j = np.unravel_index(ss, mask_img.shape)
            '''
            # Patch
            if patch_space:
                patches.append(
                    np.ravel_multi_index(
                        (i//p_sz,j//p_sz), 
                        (mask_img.shape[0]//p_sz,
                         mask_img.shape[1]//p_sz)))
            else:
            '''
            patches.append(ss)
            # Fill in the patch
            i_s = (i+np.arange(p_sz)).tolist() * p_sz
            j_s = []
            for kk in range(p_sz):
                j_s.extend([j+kk]*p_sz)
            f_idx = np.ravel_multi_index((i_s, j_s), mask_img.shape)
            flat_mask[f_idx] = 0

    # Return
    return patches


def patch_stats_img(items:list, p_sz:int=4,
        stats=['meanT', 'stdT', 'median_diff', 
                 'std_diff', 'max_diff', 'i_patch', 
                 'j_patch'],
        bias:float=0.):
    """Measure stats of patches in a single image

    Args:
        items (list): List of necesary items
            Packed this way for multi-processing
            original image, reconstructed image, mask image
            data_img, recon_img, mask_img = items
        p_sz (int, optional): Size of the patch (edge)

    Returns:
        tuple: _description_
    """

    # Unpack
    data_img, recon_img, mask_img = items
    # Find the patches
    patches = find_patches(mask_img, p_sz)

    i_patch = np.zeros(len(patches))
    j_patch = np.zeros(len(patches))

    # Build the data
    ptch_data = np.zeros(((len(patches), p_sz, p_sz)))
    ptch_recon = np.zeros(((len(patches), p_sz, p_sz)))

    # Fill me up for each patch
    for kk, patch in enumerate(patches):
        i, j = np.unravel_index(patch, mask_img.shape)
        ptch_data[kk,...] = data_img[i:i+p_sz, j:j+p_sz]
        ptch_recon[kk,...] = recon_img[i:i+p_sz, j:j+p_sz] - bias
        # Save
        i_patch[kk] = i
        j_patch[kk] = j

    # Time for stats on original data
    meanT = np.mean(ptch_data, axis=(1,2))
    stdT = np.std(ptch_data, axis=(1,2))
    T90 = np.percentile(ptch_data, 90, axis=(1,2))
    T10 = np.percentile(ptch_data, 10, axis=(1,2))
    DT = T90 - T10

    # Diff
    std_diff = np.std(ptch_data-ptch_recon, axis=(1,2))
    #mean_diff = np.mean(ptch_data-ptch_recon, axis=(1,2))
    median_diff = np.median(ptch_data-ptch_recon, axis=(1,2))
    max_diff = np.max(np.abs(ptch_data-ptch_recon), axis=(1,2))

    # Recon measurements
    T90 = np.percentile(ptch_recon, 90, axis=(1,2))
    T10 = np.percentile(ptch_recon, 10, axis=(1,2))
    DT_recon = T90 - T10

    # Generate a stat dict
    stat_dict = {}
    for istat in stats:
        stat_dict[istat] = eval(istat)

    # Return
    return stat_dict


def patchify_mask(input_mask:np.ndarray,
                  p_sz:int, rebin:tuple=None):

    if rebin is None:
        rebin = (1,1)

    # Find the patches
    mask_idx = np.where(input_mask)
    #p_idx = (mask_idx[0]//p_sz, mask_idx[1]//p_sz)
    p_idx = (mask_idx[0]//(p_sz*rebin[0]), 
             mask_idx[1]//(p_sz*rebin[1]))

    # For loop; a touch slow but not so bad
    new_mask = np.zeros((input_mask.shape[0]//rebin[0], 
                         input_mask.shape[1]//rebin[1]), 
                        dtype=int)

    for ii, jj in zip(p_idx[0], p_idx[1]):
        new_mask[ii*p_sz:(ii+1)*p_sz, 
                 jj*p_sz:(jj+1)*p_sz] = 1

    # Return
    return new_mask