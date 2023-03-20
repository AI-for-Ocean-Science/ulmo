""" Routines to analyze cutouts """
import numpy as np
import os

import h5py

from IPython import embed


def measure_bias(idx, f_orig, f_recon, f_mask, p_sz=4, nsub=1000):

    # Grab em
    orig_img = f_orig['valid'][idx,0,...]
    recon_img = f_recon['valid'][idx,0,...]
    mask_img = f_mask['valid'][idx,0,...]

    diff_true = recon_img - orig_img 

    median_bias = np.median(diff_true[np.abs(diff_true) > 0.])
    mean_bias = np.mean(diff_true[np.abs(diff_true) > 0.])

    mean_img = np.mean(orig_img[np.isclose(mask_img,0.)])

    stats = {}
    stats['median_bias'] = median_bias
    stats['mean_bias'] = mean_bias
    stats['mean_img'] = mean_img

    return stats


if __name__ == "__main__":

    preproc_path = os.path.join(os.getenv('OS_AI'), 'MAE', 'PreProc')
    recon_path = os.path.join(os.getenv('OS_AI'), 'MAE', 'Recon')
    orig_file = os.path.join(preproc_path, 'MAE_LLC_valid_nonoise_preproc.h5')
    recon_file = os.path.join(recon_path, 'mae_reconstruct_t75_p10.h5')
    mask_file = os.path.join(recon_path, 'mae_mask_t75_p10.h5')

    # Load up images
    f_orig = h5py.File(orig_file, 'r')
    f_recon = h5py.File(recon_file, 'r')
    f_mask = h5py.File(mask_file, 'r')

    for idx in range(100):
        stats = measure_bias(idx, f_orig, f_recon, f_mask, p_sz=4, nsub=1000)
        print(f"{idx:3d}: mean_bias={stats['mean_bias']:5.3f}, "+\
            f"median_bias={stats['median_bias']:5.3f} mean_img={stats['mean_img']:5.3f}")