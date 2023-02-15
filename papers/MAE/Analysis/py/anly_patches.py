""" Routines to analyze patches """
import numpy as np
import os

import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo.mae import mae_utils
from ulmo.mae import patch_analysis

from IPython import embed

def parse_metric(tbl, metric):

    if metric == 'abs_median_diff':
        values = np.abs(tbl.median_diff)
        label = r'$|\rm median\_diff |$'
    elif metric == 'median_diff':
        values = tbl.median_diff
        label = 'median_diff'
    elif metric == 'std_diff':
        values = tbl.std_diff
        label = 'rms_diff'
    elif metric == 'log10_std_diff':
        values = np.log10(tbl.std_diff)
        label = 'log10_rms_diff'
    elif metric == 'log10_stdT':
        values = np.log10(tbl.stdT)
        label = 'log10_stdT'
    else:
        raise IOError(f"bad metric: {metric}")

    # Return
    return values, label



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

    stats=['meanT', 'stdT', 'median_diff', 
           'std_diff', 'max_diff', 'i', 'j']
    map_fn = partial(patch_analysis.patch_stats_img, p_sz=p_sz,
                     stats=stats)

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
        for kk, stat_dict in enumerate(answers):
            for nn in range(nitems):
                output[i0+kk,:,nn] = stat_dict[stats[nn]]

    # Write to disk
    np.savez(outfile, data=output, items=stats)
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

    stats=['meanT', 'stdT', 'median_diff', 
           'std_diff', 'max_diff', 'i_patch', 'j_patch']
    stat_dict = patch_analysis.patch_stats_img([t_data, t_recon, t_mask], p_sz=4,
                                   stats=stats)
    '''

    # Testing full set
    anlayze_full_test(10, 20, debug=True, nsub=1000)