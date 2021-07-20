""" Module for Ulmo analysis on VIIRS 2013"""
import os
import numpy as np

import time
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import trange
import argparse


import h5py


import torch

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils

from ulmo.ssl import analysis as ssl_analysis
from ulmo.ssl.util import adjust_learning_rate
from ulmo.ssl.util import set_optimizer, save_model
from ulmo.ssl import latents_extraction

from ulmo.ssl.train_util import Params, option_preprocess
from ulmo.ssl.train_util import modis_loader, set_model
from ulmo.ssl.train_util import train_model

from IPython import embed

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("argument for training.")
    parser.add_argument("--opt_path", type=str, help="path of 'opt.json' file.")
    parser.add_argument("--func_flag", type=str, help="flag of the function to be execute: 'train' or 'evaluate'.")
        # JFH Should the default now be true with the new definition.
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    args = parser.parse_args()
    
    return args

def ssl_v2_umap(debug=False, orig=False):
    """Run a UMAP analysis on all the MODIS L2 data

    Args:
        debug (bool, optional): [description]. Defaults to False.
        orig (bool, optional): [description]. Defaults to False.
    """
    # Load table
    tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Train the UMAP

    # Split
    train = modis_tbl.pp_type == 0
    valid = modis_tbl.pp_type == 0
    y2010 = modis_tbl.pp_file == 's3://modis-l2/PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'
    valid_tbl = modis_tbl[valid & y2010].copy()
    train_tbl = modis_tbl[train & y2010].copy()

    # Latents file (subject to move)
    latents_train_file = 's3://modis-l2/SSL/SSL_v2_2012/latents/MODIS_R2019_2010_95clear_128x128_latents_std.h5'

    # Load em in
    basefile = os.path.basename(latents_train_file)
    if not os.path.isfile(basefile):
        print("Downloading latents (this is *much* faster than s3 access)...")
        ulmo_io.download_file_from_s3(basefile, latents_train_file)
        print("Done")
    hf = h5py.File(basefile, 'r')
    latents_train = hf['modis_latents_v2_train'][:]
    latents_valid = hf['modis_latents_v2_valid'][:]
    print("Latents loaded")

    # Table (useful for valid only)
    modis_tbl = ulmo_io.load_main_table('s3://modis-l2/Tables/MODIS_L2_std.parquet')

    # Check
    assert latents_valid.shape[0] == len(valid_tbl)

    # Stack em
    latents = np.concatenate([latents_train, latents_valid])
    _, _, latents_mapping = ssl_analysis.latents_umap(
        latents, np.arange(latents_train.shape[0]), 
        latents_train.shape[0]+np.arange(latents_valid.shape[0]),
        valid_tbl, debug=False)

    # Loop on em all
    latent_files = ulmo_io.list_of_bucket_files('modis-l2',
                                                prefix='SSL/SSL_v2_2012/latents/')

    for latents_file in latent_files:
        basefile = os.path.basename(latents_file)
        # Download?
        if not os.path.isfile(basefile):
            print(f"Downloading {latents_file} (this is *much* faster than s3 access)...")
            ulmo_io.download_file_from_s3(basefile, latents_train_file)
            print("Done")
        # 
        hf = h5py.File(basefile, 'r')
        latents_train = hf['modis_latents_v2_train'][:]
        latents_valid = hf['modis_latents_v2_valid'][:]


    # Vet
    assert cat_utils.vet_main_table(valid_tbl, cut_prefix='modis_')


def main_train(opt_path: str):
    """Train the model

    Args:
        opt_path (str): Path + filename of options file
    """
    # loading parameters json file
    opt = Params(opt_path)
    opt = option_preprocess(opt)

    # build data loader
    train_loader = modis_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt, cuda_use=opt.cuda_use)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    # training routine
    for epoch in trange(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train_model(train_loader, model, criterion, 
                           optimizer, epoch, opt, cuda_use=opt.cuda_use)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            # Save locally
            save_file = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
            save_model(model, optimizer, opt, epoch, save_file)
            # Save to s3
            s3_file = os.path.join(
                opt.s3_outdir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            ulmo_io.upload_file_to_s3(save_file, s3_file)

    # save the last model local
    save_file = 'last.pth'
    save_model(model, optimizer, opt, opt.epochs, save_file)
    # Save to s3
    s3_file = os.path.join(opt.s3_outdir, 'last.pth')
    ulmo_io.upload_file_to_s3(save_file, s3_file)

        
def main_evaluate(opt_path, model_file, 
                  preproc='_std', debug=False):
    """
    This function is used to obtain the latents of the trained models
    for all of MODIS

    Args:
        opt_path: (str) option file path.
        model_file: (str) s3 filename
        preproc: (str, optional)
    """
    opt = option_preprocess(Params(opt_path))

    model_base = os.path.basename(model_file)
    ulmo_io.download_file_from_s3(model_base, model_file)
    
    # Data files
    all_pp_files = ulmo_io.list_of_bucket_files(
        'modis-l2', 'PreProc')
    pp_files = []
    for ifile in all_pp_files:
        if preproc in ifile:
            pp_files.append(ifile)

    # Loop on files
    key_train, key_valid = "train", "valid"
    if debug:
        pp_files = pp_files[0:1]
    for ifile in pp_files:
        print("Working on ifile")
        data_file = os.path.basename(ifile)
        if not os.path.isfile(data_file):
            ulmo_io.download_file_from_s3(data_file, 
            f's3://modis-l2/PreProc/{data_file}')

        # Read
        with h5py.File(data_file, 'r') as file:
            if 'train' in file.keys():
                train=True
            else:
                train=False

        # Setup
        latents_file = data_file.replace('_preproc', '_latents')
        latents_path = os.path.join(opt.latents_folder, latents_file) 
        if train: 
            print("Starting train evaluation")
            latents_extraction.model_latents_extract(opt, data_file, 
                'train', model_base, latents_file, key_train)
            print("Extraction of Latents of train set is done.")
        print("Starting valid evaluation")
        latents_extraction.model_latents_extract(opt, data_file, 
                'valid', model_base, latents_file, key_valid)
        print("Extraction of Latents of valid set is done.")

        # Push to s3
        print("Uploading to s3..")
        ulmo_io.upload_file_to_s3(latents_file, latents_path)

        # Remove data file
        os.remove(data_file)
        print(f'{data_file} removed')
        
if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    
    # run the 'main_train()' function.
    if args.func_flag == 'train':
        print("Training Starts.")
        main_train(args.opt_path)
        print("Training Ends.")
    
    # run the "main_evaluate()" function.
    if args.func_flag == 'evaluate':
        print("Evaluation Starts.")
        main_evaluate(args.opt_path, 
                      's3://modis-l2/SSL/SSL_v2_2012/last.pth',
                      debug=args.debug)
        print("Evaluation Ends.")

    # run the umap
    if args.func_flag == 'umap':
        print("UMAP Starts.")
        ssl_v2_umap()
        print("UMAP Ends.")
