""" Routines to analyze patches """
import numpy as np

import h5py

from IPython import embed

def find_patches(mask_img, p_sz:int):

    flat_mask = mask_img.flatten().astype(int)
    patches = []
    for ss in range(mask_img.size):
        if flat_mask[ss] == 0:
            patches.append(ss)
            # Fill in the patch
            i, j = np.unravel_index(ss, mask_img.shape)
            #import pdb; pdb.set_trace()
            i_s = (i+np.arange(p_sz)).tolist() * p_sz
            j_s = []
            for kk in range(p_sz):
                j_s.extend([j+kk]*p_sz)
            f_idx = np.ravel_multi_index((i_s, j_s), mask_img.shape)
            flat_mask[f_idx] = 1

    # Return
    return patches

def patch_stats_img(data_img, recon_img, mask_img, 
                    p_sz:int):

    # Find the patches
    patches = find_patches(mask_img, p_sz)

    # Build the data
    ptch_data = np.zeros(((len(patches), p_sz, p_sz)))
    ptch_recon = np.zeros(((len(patches), p_sz, p_sz)))

    # Fill me up for each patch
    for kk, patch in enumerate(patches):
        i, j = np.unravel_index(patch, mask_img.shape)
        ptch_data[kk,...] = data_img[i:i+p_sz, j:j+p_sz]
        ptch_recon[kk,...] = recon_img[i:i+p_sz, j:j+p_sz]

    # Time for stats
    meanT = np.mean(ptch_data, axis=(1,2))
    stdT = np.std(ptch_data, axis=(1,2))

    # Diff
    std_diff = np.std(ptch_data-ptch_recon, axis=(1,2))
    mean_diff = np.mean(ptch_data-ptch_recon, axis=(1,2))
    embed(header='57 of anly_patches.py')

if __name__ == "__main__":
    # Testing

    f_mask = h5py.File('mae_mask_t75_p75_small.h5', 'r')
    f_recon = h5py.File('mae_reconstruct_t75_p75_small.h5', 'r')
    f_data = h5py.File('mae_reconstruct_t75_p75_small.h5', 'r')

    idx = 0
    t_mask = f_mask['valid'][idx, 0, ...]
    t_recon = f_recon['valid'][idx, 0, ...]
    t_data = f_data['valid'][idx, 0, ...]

    patch_stats_img(t_data, t_recon, t_mask, 4)