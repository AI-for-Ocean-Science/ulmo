""" Routines to analyze patches """
import numpy as np

import h5py


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
        10, 20, debug=True, nsub=1000)