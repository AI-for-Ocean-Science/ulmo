""" Utility methods for Enki """

from pkg_resources import resource_filename

import os
import numpy as np

import pandas

def load_bias(tp:tuple=None, bias_path:str=None, dataset:str='LLC2_nonoise'):
    """ Load the bias values

    Args:
        tp (tuple, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Name
    if bias_path is None:
        if dataset == 'VIIRS':
            dataset = 'LLC2_nonoise'
        bias_path = os.path.join(resource_filename('ulmo', 'runs'), 
            'Enki', 'Masters', f'enki_bias_{dataset}.csv')
    # Load
    print(f"Loading bias table from {bias_path}")
    bias = pandas.read_csv(bias_path)
    # Value?
    if tp is not None:
        bias = float(bias[(bias.t==tp[0]) & (bias.p==tp[1])]['median'])
    return bias

def img_filename(t_per:int, p_per:int,
                 mae_img_path:str=None,
                 local:bool=False,
                 dataset:str='LLC'):
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
        dataset (str, optional):
            Dataset name
    Returns:
        str: filename with s3 path
    
    """
    # Path
    if mae_img_path is None:
        if local:
            if dataset == 'LLC':
                root = 'mae'
                dpath = os.path.join(os.getenv('OS_OGCM'), 'LLC')
            elif dataset == 'LLC2_nonoise':
                root = 'enki'
                dpath = os.path.join(os.getenv('OS_OGCM'), 'LLC')
            elif dataset == 'LLC2_noise':
                root = 'enki_noise'
                dpath = os.path.join(os.getenv('OS_OGCM'), 'LLC')
            elif dataset == 'LLC2_noise02':
                root = 'enki_noise02'
                dpath = os.path.join(os.getenv('OS_OGCM'), 'LLC')
            elif dataset == 'VIIRS':
                root = 'VIIRS_100clear'
                dpath = os.path.join(os.getenv('OS_SST'), 'VIIRS')
            mae_img_path = os.path.join(dpath, 'Enki', 'Recon')
        else:
            if dataset != 'LLC':
                raise NotImplementedError("Only LLC for now")
            mae_img_path = 's3://llc/mae/Recon'
            root = 'mae'
    # Base
    base_name = f'{root}_reconstruct_t{t_per:02d}_p{p_per:02d}.h5'

    # Finish
    img_file = os.path.join(mae_img_path, base_name)

    # Return
    return img_file

def mask_filename(t_per:int, p_per:int,
                 mae_img_path:str=None,
                 local:bool=False,
                 dataset:str='LLC'):
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
        dataset (str, optional):
            Dataset name
    Returns:
        str: filename with s3 path
    
    """
    recon_file = img_filename(t_per, p_per, mae_img_path=mae_img_path, local=local,
                              dataset=dataset)
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

def set_files(dataset:str, t:int, p:int):
    """ Set the files for a given dataset, t, p

    Args:
        dataset (str): 
            Dataset name ['VIIRS', 'LLC', 'LLC2', 'LLC2_nonoise']
        t (int): 
            Train percentile
        p (int): 
            Patch percentile

    Raises:
        ValueError: _description_

    Returns:
        tuple: 4 str object -- tbl_file, orig_file, recon_file, mask_file
    """
    sst_path = os.getenv('OS_SST')
    enki_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki')

    if dataset == 'VIIRS':
        tbl_file = os.path.join('s3://viirs', 'Tables', 
                                'VIIRS_all_100clear_std.parquet')
        orig_file = os.path.join(sst_path, 'VIIRS', 
                                          'PreProc', 'VIIRS_all_100clear_preproc.h5')
    elif dataset == 'LLC':
        tbl_file = 's3://llc/mae/Tables/MAE_LLC_valid_nonoise.parquet'
        orig_file = os.path.join(enki_path, 'PreProc', 
                                 'MAE_LLC_valid_nonoise_preproc.h5')
    elif dataset == 'LLC2_nonoise':
        tbl_file = 's3://llc/mae/Tables/Enki_LLC_valid_nonoise.parquet'
        orig_file = os.path.join(enki_path, 'PreProc', 
                                 'Enki_LLC_valid_nonoise_preproc.h5')
    elif dataset == 'LLC2_noise':
        tbl_file = 's3://llc/mae/Tables/Enki_LLC_valid_noise.parquet'
        orig_file = os.path.join(enki_path, 'PreProc', 
                                 'Enki_LLC_valid_noise_preproc.h5')
    elif dataset == 'LLC2_noise02':
        tbl_file = 's3://llc/mae/Tables/Enki_LLC_valid_noise02.parquet'
        orig_file = os.path.join(enki_path, 'PreProc', 
                                 'Enki_LLC_valid_noise02_preproc.h5')
    else:
        raise ValueError("Bad dataset")

    recon_file = img_filename(t,p, local=True, dataset=dataset)
    mask_file = mask_filename(t,p, local=True, dataset=dataset)

    return tbl_file, orig_file, recon_file, mask_file
