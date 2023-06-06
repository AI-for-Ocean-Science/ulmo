""" Methods related to the bias correction"""

import os

from pkg_resources import resource_filename

import pandas

import numpy as np
import h5py

def measure_bias(f_orig:h5py.File, f_recon:h5py.File, nsamples:int=None):
    """ Measure the bias

    Args:
        f_orig (h5py.File): 
            Pointer to the original images
        f_recon (h5py.File): 
            Pointer to the reconstructed images
        nsamples (int, optional): 
            Number of samples to use

    Returns:
        tuple: np.array's of median and mean biases
    """

    # Values
    median_biases = []
    mean_biases = []

    if nsamples is None:
        nsamples = f_orig['valid'].shape[0]

    for idx in range(nsamples):
        orig_img = f_orig['valid'][idx,0,...]
        recon_img = f_recon['valid'][idx,0,...]

        diff_true = recon_img - orig_img 

        median_bias = np.median(diff_true[np.abs(diff_true) > 0.])
        mean_bias = np.mean(diff_true[np.abs(diff_true) > 0.])
        #mean_img = np.mean(orig_img[np.isclose(mask_img,0.)])
        # Save
        median_biases.append(median_bias)
        mean_biases.append(mean_bias)

    # Return
    return np.array(median_biases), np.array(mean_biases)

def load_bias_table():
    """ Simple method to load up the bias values

    Returns:
        pandas.DataFrame: table of bias values
    """
    bias_file = os.path.join(
            resource_filename('ulmo', 'runs'),
            'Enki', 'Masters', 'enki_bias_LLC.csv')
    bias = pandas.read_csv(bias_file)
    return bias