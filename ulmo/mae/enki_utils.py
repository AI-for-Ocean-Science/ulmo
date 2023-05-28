""" Utility methods for Enki """
import os
import numpy as np

import pandas

def img_filename(t_per:int, p_per:int,
                 mae_img_path:str=None,
                 local:bool=False):
    """Generate the image filename from the
    percentiles
    Args:
        t_per (int):
            Patch percentile of the model used 
        p_per (int):
            Patch percentile of the data
        mae_img_path (str, optional):
            path to the image file
        local (bool, optional):
            Use local path?
    Returns:
        str: filename with s3 path
    
    """
    # Path
    if mae_img_path is None:
        if local:
            mae_img_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'Recon')
        else:
            mae_img_path = 's3://llc/mae/Recon'
    # Base
    base_name = f'mae_reconstruct_t{t_per:02d}_p{p_per:02d}.h5'

    # Finish
    img_file = os.path.join(mae_img_path, base_name)

    # Return
    return img_file

def mask_filename(t_per:int, p_per:int,
                 mae_img_path:str=None,
                 local:bool=False):
    """Generate the image filename from the
    percentiles
    Args:
        t_per (int):
            Patch percentile of the model used 
        p_per (int):
            Patch percentile of the data
        mae_img_path (str, optional):
            s3 path to the image file
        local (bool, optional):
            Use local path?
    Returns:
        str: filename with s3 path
    
    """
    recon_file = img_filename(t_per, p_per, mae_img_path=mae_img_path, local=local)
    mask_file = recon_file.replace('reconstruct', 'mask')

    return mask_file

def parse_enki_file(ifile:str):
    """ Grab the train and patch percentages from the input filename

    Args:
        img_file (str): 
            Assumes mae_reconstruct_tXX_pXX.h5 format

    Returns:
        tuple: train and patch percentages (str,str`)
    """

    prs = os.path.basename(ifile).split('_')

    # Train %
    assert prs[2][0] == 't'
    t_per = prs[2][1:]

    # Patch %
    assert prs[3][0] == 'p'
    p_per = prs[3][1:3]

    return t_per, p_per


def parse_metric(tbl:pandas.DataFrame, metric:str):
    """ Parse the metric

    Args:
        tbl (pandas.DataFrame): 
            Table of patch statistics
        metric (str): 
            Metric to parse

    Raises:
        IOError: _description_

    Returns:
        tuple: values (np.ndarray), label (str)
    """

    if metric == 'abs_median_diff':
        values = np.abs(tbl.median_diff)
        label = r'$|\rm median\_diff |$'
    elif metric == 'median_diff':
        values = tbl.median_diff
        label = 'median_diff'
    elif metric == 'std_diff':
        values = tbl.std_diff
        label = 'RMSE'
    elif metric == 'log10_std_diff':
        values = np.log10(tbl.std_diff)
        label = r'$\log_{10} \, \rm RMSE$'
    elif metric == 'log10_stdT':
        values = np.log10(tbl.stdT)
        label = r'$\log_{10} \, \sigma_{T}$'
    else:
        raise IOError(f"bad metric: {metric}")

    # Return
    return values, label