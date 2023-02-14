""" Routines to analyze patches """
import numpy as np
import os

import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo.mae import mae_utils

from IPython import embed

def find_patches(mask_img, p_sz:int):

    flat_mask = mask_img.flatten().astype(int)
    patches = []
    for ss in range(mask_img.size):
        if flat_mask[ss] == 1:
            patches.append(ss)
            # Fill in the patch
            i, j = np.unravel_index(ss, mask_img.shape)
            #import pdb; pdb.set_trace()
            i_s = (i+np.arange(p_sz)).tolist() * p_sz
            j_s = []
            for kk in range(p_sz):
                j_s.extend([j+kk]*p_sz)
            f_idx = np.ravel_multi_index((i_s, j_s), mask_img.shape)
            flat_mask[f_idx] = 0

    # Return
    return patches

def patch_stats_img(items:list, p_sz:int=4):

    # Unpack
    data_img, recon_img, mask_img = items
    # Find the patches
    patches = find_patches(mask_img, p_sz)

    i_patches = np.zeros(len(patches))
    j_patches = np.zeros(len(patches))

    # Build the data
    ptch_data = np.zeros(((len(patches), p_sz, p_sz)))
    ptch_recon = np.zeros(((len(patches), p_sz, p_sz)))

    # Fill me up for each patch
    for kk, patch in enumerate(patches):
        i, j = np.unravel_index(patch, mask_img.shape)
        ptch_data[kk,...] = data_img[i:i+p_sz, j:j+p_sz]
        ptch_recon[kk,...] = recon_img[i:i+p_sz, j:j+p_sz]
        # Save
        i_patches[kk] = i
        j_patches[kk] = j

    # Time for stats
    meanT = np.mean(ptch_data, axis=(1,2))
    stdT = np.std(ptch_data, axis=(1,2))

    # Diff
    std_diff = np.std(ptch_data-ptch_recon, axis=(1,2))
    #mean_diff = np.mean(ptch_data-ptch_recon, axis=(1,2))
    median_diff = np.median(ptch_data-ptch_recon, axis=(1,2))
    max_diff = np.max(np.abs(ptch_data-ptch_recon), axis=(1,2))

    # Return
    stats = ['meanT', 'stdT', 'median_diff', 'std_diff', 'max_diff', 'i', 'j']
    return meanT, stdT, median_diff, std_diff, max_diff, i_patches, j_patches, stats

def anlayze_full_test(t_per:int, p_per:int, 
    orig_file='MAE_LLC_valid_nonoise_preproc.h5',
    nsub:int=1000, n_cores:int=4, p_sz:int=4, debug:bool=False):
    # Grab the files (if not local)

    # Reconstruction file
    recon_file = mae_utils.img_filename(t_per, p_per)
    base_recon = os.path.basename(recon_file)
    if not os.path.isfile(base_recon):
        embed(header='download it!')
    # Mask file
    mask_file = recon_file.replace('reconstruct', 'mask')
    base_mask = os.path.basename(mask_file)
    if not os.path.isfile(base_mask):
        embed(header='download it!')

    # Outfile
    outfile = base_mask.replace('mask', 'patches')
    outfile = outfile.replace('.h5', '.npz')

    # Load up
    f_mask = h5py.File(base_mask, 'r')
    f_recon = h5py.File(base_recon, 'r')
    f_orig = h5py.File(orig_file, 'r')
    nimages = f_mask['valid'].shape[0]

    if debug:
        nimages = 10000

    map_fn = partial(patch_stats_img, p_sz=p_sz)

    # Run one to get the number of patches and number of items?
    npatches = 52  
    nitems = 7

    # Output array
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
        for kk, item in enumerate(answers):
            for nn in range(nitems):
                output[i0+kk,:,nn] = item[nn]

    # Write to disk
    np.savez(outfile, data=output, items=item[-1])
    print(f'Wrote: {outfile}')

if __name__ == "__main__":

    '''
    # Testing single image
    f_mask = h5py.File('mae_mask_t75_p75_small.h5', 'r')
    f_recon = h5py.File('mae_reconstruct_t75_p75_small.h5', 'r')
    f_data = h5py.File('mae_reconstruct_t75_p75_small.h5', 'r')

    idx = 0
    t_mask = f_mask['valid'][idx, 0, ...]
    t_recon = f_recon['valid'][idx, 0, ...]
    t_data = f_data['valid'][idx, 0, ...]

    patch_stats_img(t_data, t_recon, t_mask, 4)
    '''

    # Testing full set
    anlayze_full_test(10, 20, debug=True, nsub=1000)