from __future__ import print_function

import time
import os
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import trange
import argparse


import torch

from ulmo import io as ulmo_io
from ulmo.llc import extract 

from ulmo.ssl.util import adjust_learning_rate
from ulmo.ssl.util import set_optimizer, save_model

from ulmo.ssl.train_util import Params, option_preprocess
from ulmo.ssl.train_util import modis_loader_v2, set_model
from ulmo.ssl.train_util import train_model

from ulmo.utils import catalog as cat_utils

from ulmo.ssl import analysis as ssl_analysis

from IPython import embed

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("argument for training.")
    #parser.add_argument("--opt_path", type=str, help="path of 'opt.json' file.")
    parser.add_argument("--func_flag", type=str, 
                        help="flag of the function to be execute: 'extract_curl'.")
    args = parser.parse_args()
    
    return args

def extract_curl(debug=True):

    orig_tbl_file = 's3://llc/Tables/test_noise_modis2012.parquet'
    tbl_file = 's3://llc/Tables/modis2012_kin_curl.parquet'
    root_file = 'LLC_modis2012_curl_preproc.h5'
    llc_table = ulmo_io.load_main_table(orig_tbl_file)

    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    # Run it
    llc_table = extract.velocity_field(llc_table, 'curl',
                                pp_local_file, 
                                'llc_std',
                                s3_file=pp_s3_file,
                                debug=debug)
    # Vet
    assert cat_utils.vet_main_table(llc_table, cut_prefix='modis_')

    # Final write
    if not debug:
        ulmo_io.write_main_table(llc_table, tbl_file)
    


if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    
    # run the 'main_train()' function.
    if args.func_flag == 'extract_curl':
        print("Extraction starts.")
        extract_curl(debug=False)
        print("Extraction Ends.")
    
    # run the "main_evaluate()" function.
    if args.func_flag == 'evaluate':
        print("Evaluation Starts.")
        main_evaluate(args.opt_path)
        print("Evaluation Ends.")

    # run the "main_evaluate()" function.
    if args.func_flag == 'umap':
        print("Generating the umap")
        generate_umap()