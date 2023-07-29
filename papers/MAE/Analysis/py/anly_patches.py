""" Routines to analyze patches """
import numpy as np

import h5py


from ulmo.mae import patch_analysis

from IPython import embed





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
           'std_diff', 'max_diff', 'i_patch', 'j_patch', 'DT']
    stat_dict = patch_analysis.patch_stats_img([t_data, t_recon, t_mask], p_sz=4,
                                   stats=stats)
    '''

    # Testing full set
    patch_analysis.anlayze_full_test(
        's3://llc/mae/Recon/mae_reconstruct_t10_p20.h5')
        