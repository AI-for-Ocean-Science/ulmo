""" Module for Ulmo analysis on VIIRS 2013"""
import os
import glob
from h5py._hl import base
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

def ssl_test_eval_orig(debug=False, orig=False):
    model_path = './'
    model_name = "last.pth"
    # Load options
    opt_file = os.path.join(resource_filename('ulmo', 'runs'),
                            'SSL', 'First','experiments', 
                            'base_modis_model', 'opts.json')
    opt = Params(opt_file)
    opt = option_preprocess(opt)
    base_file = 'test_orig.h5'
    
    # Load
    hf = h5py.File(base_file, 'r')
    modis_data = hf['train'][:]

    # Output
    model_path_title = os.path.join(model_path, model_name)
    model_name = model_name.split('.')[0]
    latents_path = 'orig_latents.h5'
    save_key = 'modis_latents'
    
    # Run
    latents_extraction.orig_latents_extract(opt, modis_data,
                                             model_path_title, 
                                             latents_path, 
                                             save_key)

def ssl_eval_2010(dataset, debug=False, orig=False):
    s3_model_path = 's3://modis-l2/modis_simclr_base_model/SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/last.pth'
    ulmo_io.download_file_from_s3(os.path.basename(s3_model_path), s3_model_path)
    model_path = './'
    model_name = "last.pth"

    # Load options
    opt_file = os.path.join(resource_filename('ulmo', 'runs'),
                            'SSL', 'First','experiments', 
                            'base_modis_model', 'opts.json')
    opt = Params(opt_file)
    opt = option_preprocess(opt)

    # Load the data
    print("Grabbing MODIS [if needed]")
    if orig:
        modis_dataset_path = "s3://modis-l2/PreProc/MODIS_2010_95clear_128x128_inpaintT_preproc_0.8valid.h5"
    else:
        modis_dataset_path = "s3://modis-l2/PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5"
    base_file = os.path.basename(modis_dataset_path)
    if not os.path.isfile(base_file):
        ulmo_io.download_file_from_s3(base_file, modis_dataset_path)

    # Output
    model_path_title = os.path.join(model_path, model_name)
    model_name = model_name.split('.')[0]
    if orig:
        latents_path = f'MODIS_orig_2010_{dataset}_{model_name}.h5'
    else:
        latents_path = f'MODIS_2010_{dataset}_{model_name}.h5'
    save_key = 'modis_latents'
    
    # Run
    latents_extraction.model_latents_extract(opt, base_file, dataset,
                                             model_path_title, latents_path, 
                                             save_key)
    # Push to s3                                            
    s3_outfile = 's3://modis-l2/modis_latents_simclr/'+latents_path
    ulmo_io.upload_file_to_s3(latents_path, s3_outfile)



def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Evaluate the MODIS training data from 2010
    if flg & (2**0):
        ssl_eval_2010('train')

    # Evaluate the MODIS valid data from 2010
    if flg & (2**1):
        ssl_eval_2010('valid')

    # Evaluate the MODIS valid data from 2010
    if flg & (2**2):
        ssl_test_eval_orig()

    # Evaluate the MODIS training data from 2010
    if flg & (2**3):
        ssl_eval_2010('train', orig=True)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- MODIS 2010 training
        #flg += 2 ** 1  # 2 -- MODIS 2010 valid
        #flg += 2 ** 2  # 4 -- Debuggin
        flg += 2 ** 3  # 8 -- re-run orig
    else:
        flg = sys.argv[1]

    main(flg)