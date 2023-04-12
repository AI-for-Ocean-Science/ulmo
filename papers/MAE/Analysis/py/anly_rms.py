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
    rms = diff.mean()
    rms = np.sqrt(rms)
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
    avgs.to_parquet('valid_avg_rms.parquet')
    return rms

# Calculates rms using table.
def calculate_avg_rms(table, errors, start, end, num_imgs):
    """
    Calculate rms of a range on the LL table. Can calculate rms for the full dataset as well
    table:    LL table (sorted if checking rms of batches)
    errors:   np array of errors (same order as original file)
    start:    Iteration start
    end:      Iteration end
    num_imgs: number of images in the current batch
    """
    summ = 0
    for i in range(start, end):
        idx = int(table.iloc[i]['pp_idx'])
        summ = summ + errors[idx]
    
    avg = summ/num_imgs
    return avg
        
def calc_batch_rms(table, errors, batch_percent):
    """
    Calculates rms in batches. Handles extra by adding them to final batch
    so pick reasonable batch sizes that won't leave a lot of extra 
    table:   LL table (sorted)
    errors:  np array of errors (same order as original file)
    batch_percent: batch_percent to batch
    """
    # Uncomment this when working with not broken files
    # assert len(table.index) == len(errors)
    num_imgs = len(table.index)
    batch_size = int(num_imgs*batch_percent/100) # size of batch
    num_batches = num_imgs // batch_size # batches to run excluding final batch
    final_batch = num_imgs-batch_size*(num_batches-1)

    print('number of images:', num_imgs,'\nnumber of batches:', num_batches,
          '\nbatch size:', batch_size, '\nfinal batch:', final_batch)
    rms = np.empty(num_batches, dtype=np.float64)
    for batch in range(num_batches-1):
        start = batch*batch_size
        end = start + batch_size
        rms[batch] = calculate_rms(table, errors, start, end, batch_size)
        #print('average of batch',batch+1,'is',rms[batch])
    
    rms[num_batches-1] = calculate_avg_rms(table, errors, batch_size*(num_batches-1), num_imgs, final_batch)
    #print('average of batch',num_batches,'is',rms[num_batches-1])
    
    return rms


def create_table(outfile='valid_avg_rms.csv',
                 LL_filepath='MAE_LLC_valid_nonoise.parquet', 
                 rms_filepath='valid_rms.parquet'):
    # load tables
    table = pd.read_parquet(LL_filepath, engine='pyarrow')
    table = table[table['LL'].notna()]
    table = table.sort_values(by=['LL'])
    rms = pd.read_parquet(rms_filepath, engine='pyarrow')
    
    # calculate median LL
    x = ["" for i in range(10)]
    for i in range(10):
        start = table.iloc[i*65578]['LL']
        end = table.iloc[(i+1)*65578]['LL']
        avg = (start + end)/2
        x[i] = str(int(avg))
    
    avgs = pd.DataFrame(x, columns=['median_rms'])

    # calculate batch averages
    models = [10, 35, 75]
    masks = [10, 20, 30, 40, 50]
    for t in models:
        for p in masks:
            index = 'rms_t{}_p{}'.format(t, p)
            avg_rms = calc_batch_rms(table, rms[index], p)
            avgs[index] = avg_rms
        
    avgs.to_csv(outfile)

# diff = calc_diff(orig_file,recon_file, mask_file)