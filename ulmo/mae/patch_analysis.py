""" Module for methods related to patch analysis.
Mainly to assess performance of the MAE"""

import numpy as np

from IPython import embed

# TODO -- Consider using jit on the following method
def find_patches(mask_img, p_sz:int):
    """ Simple algorithm to find the patches
    in a masked MAE image

    It assumes they are square and whole

    Args:
        mask_img (np.ndarray): Masked image
        p_sz (int): Size of the patch (edge)

    Returns:
        list: Ravel'd index of the patches
    """

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


def patch_stats_img(items:list, p_sz:int=4,
        stats=['meanT', 'stdT', 'median_diff', 
                 'std_diff', 'max_diff', 'i', 'j']):
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
        ptch_recon[kk,...] = recon_img[i:i+p_sz, j:j+p_sz]
        # Save
        i_patch[kk] = i
        j_patch[kk] = j

    # Time for stats
    meanT = np.mean(ptch_data, axis=(1,2))
    stdT = np.std(ptch_data, axis=(1,2))

    # Diff
    std_diff = np.std(ptch_data-ptch_recon, axis=(1,2))
    #mean_diff = np.mean(ptch_data-ptch_recon, axis=(1,2))
    median_diff = np.median(ptch_data-ptch_recon, axis=(1,2))
    max_diff = np.max(np.abs(ptch_data-ptch_recon), axis=(1,2))

    # Generate a stat dict
    stat_dict = {}
    for istat in stats:
        stat_dict[istat] = eval(istat)

    # Return
    return stat_dict
