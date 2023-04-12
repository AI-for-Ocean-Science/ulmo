import sys
import os
import requests
import time

import torch
import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.ticker as ticker

from ulmo.plotting import plotting

from ulmo.mae import mae_utils
from ulmo import io as ulmo_io
from scipy.sparse import csc_matrix

from IPython import embed

def rms_single_img(orig_img, recon_img, mask_img):
    """ Calculate rms of a single image (ignore edges)
    orig_img:  original img (64x64)
    recon_img: reconstructed image (64x64)
    mask_img:  mask of recon_image (64x64)
    """
    # remove edges
    orig_img  = orig_img[4:-4, 4:-4]
    recon_img = recon_img[4:-4, 4:-4]
    mask_img  = mask_img[4:-4, 4:-4]
    
    # Find i,j positions from mask
    mask_sparse = csc_matrix(mask_img)
    mask_i,mask_j = mask_sparse.nonzero()
    
    # Find differences
    diff = np.zeros(len(mask_i))
    for idx, (i, j) in enumerate(zip(mask_i, mask_j)):
        diff[idx] = orig_img[i,j] - recon_img[i,j]
    
    #embed(header='44 of anly_rms.py')
    diff = np.square(diff)
    mse = diff.mean()
    rms = np.sqrt(mse)
    return rms

def calc_diff(orig_file, recon_file, mask_file,
              outfile='valid_rms.parquet'):
    """
    Calculate the differences (error) between images, and save to file
    orig_file:  original images
    recon_file: reconstructed images
    mask_file:  mask file for the reconstructed images
    outfile:    file to save
    """
    t0 = time.time()
    f_og = h5py.File(orig_file, 'r')
    f_re = h5py.File(recon_file, 'r')
    f_ma = h5py.File(mask_file, 'r')
    
    num_imgs = f_og['valid'].shape[0]
    rms = np.empty(num_imgs, dtype=np.float64)
    for i in range(num_imgs):
        orig_img = f_og['valid'][i,0,...]
        recon_img = f_re['valid'][i,0,...]
        mask_img = f_ma['valid'][i,0,...]
        
        rms[i] = rms_single_img(orig_img, recon_img, mask_img)
        if i%10000 == 0:
                print(i, rms[i])
    t1 = time.time()
    total = t1-t0
    print(total)   # print final time
    
    rms = pd.read_parquet('valid_rms.parquet', engine='pyarrow')
    t, p = parse_mae_img_file(recon_file)
    rms['rms_t{}_p{}'.format(t, p)]=rms
    np.save('differences_t10_p40.npy', rms, allow_pickle=False)
    return rms

# diff = calc_diff(orig_file,recon_file, mask_file)