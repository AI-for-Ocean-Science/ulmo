""" Module for Ulmo analysis on VIIRS 2013"""
import os
import glob
import numpy as np
import subprocess 
from pkg_resources import resource_filename

import pandas
import h5py
from skimage.restoration import inpaint 

from sklearn.utils import shuffle

from ulmo import io as ulmo_io

from ulmo.ssl.my_util import Params, option_preprocess
from ulmo.ssl import latents_extraction

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import subprocess
from tqdm import tqdm

from IPython import embed

tbl_file_2013 = 's3://viirs/Tables/VIIRS_2013_std.parquet'
s3_bucket = 's3://viirs'

def ssl_eval_train_2010(debug=False):
    #model_path = "./experiments/base_modis_model/SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_1_cosine_warm/" \
    #         "SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm"
    #model_path = 's3://modis-l2/modis_simclr_base_model/SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/'
    model_path = './'
    model_name = "last.pth"

    opt_file = os.path.join(resource_filename('ulmo', 'runs'),
                            'SSL', 'First','experiments', 
                            'base_modis_model', 'opts.json')
    opt = Params(opt_file)
    opt = option_preprocess(opt)

    # Load the data
    print("Loading MODIS")
    #modis_dataset_path = "s3://modis-l2/PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5"
    print("Grabbing MODIS")
    modis_dataset_path = "s3://modis-l2/PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5"
    base_file = os.path.basename(modis_dataset_path)
    if not os.path.isfile(base_file):
        ulmo_io.download_file_from_s3(base_file, modis_dataset_path)
    #with ulmo_io.open(modis_dataset_path, 'rb') as f:
    #    hf = h5py.File(f, 'r')
    #    dataset_train = hf['train'][:]
    #print("Loaded MODIS")

    save_path = './experiments/modis_latents/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    model_path_title = os.path.join(model_path, model_name)
    model_name = model_name.split('.')[0]
    latents_path = os.path.join(save_path, f'MODIS_2010_train_{model_name}.h5')
    save_key = 'modis_latents'
    
    latents_extraction.model_latents_extract(opt, base_file, 'train', 
                                             model_path_title, latents_path, 
                                             save_key)

def ssl_eval_valid_2010(debug=False):
    #model_path = "./experiments/base_modis_model/SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_1_cosine_warm/" \
    #         "SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm"
    #model_path = 's3://modis-l2/modis_simclr_base_model/SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/'

    # TODO -- Figure out how to load from s3.  Or, at worst, copy locally and then run with it
    #s3 get s3://modis-l2/modis_simclr_base_model/SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/last.pth0
    model_path = './'
    model_name = "last.pth"

    opt_file = os.path.join(resource_filename('ulmo', 'runs'),
                            'SSL', 'First','experiments', 
                            'base_modis_model', 'opts.json')
    opt = Params(opt_file)
    opt = option_preprocess(opt)

    # Load the data
    print("Grabbing MODIS")
    modis_dataset_path = "s3://modis-l2/PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5"
    base_file = os.path.basename(modis_dataset_path)
    if not os.path.isfile(base_file):
        ulmo_io.download_file_from_s3(base_file, modis_dataset_path)
    hf = h5py.File(base_file, 'r')
    dataset_train = hf['valid'][:]
    hf.close()
    print("Loaded MODIS")

    save_path = './experiments/modis_latents/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    model_path_title = os.path.join(model_path, model_name)
    model_name = model_name.split('.')[0]
    latents_path = os.path.join(save_path, f'MODIS_2010_valid_{model_name}.h5')
    save_key = 'modis_latents'
    
    latents_extraction.model_latents_extract(opt, dataset_train, 
                                             model_path_title, latents_path, 
                                             save_key)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Evaluate the MODIS training data from 2010
    if flg & (2**0):
        ssl_eval_train_2010(debug=False)

    # Evaluate the MODIS valid data from 2010
    if flg & (2**1):
        ssl_eval_valid_2010()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- MODIS 2010 training
        #flg += 2 ** 1  # 1 -- MODIS 2010 valid
    else:
        flg = sys.argv[1]

    main(flg)