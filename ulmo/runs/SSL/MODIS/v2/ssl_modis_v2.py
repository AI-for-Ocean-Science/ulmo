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

from ulmo.ssl import analysis as ssl_analysis
from ulmo.ssl.util import adjust_learning_rate
from ulmo.ssl.util import set_optimizer, save_model

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
    # Latents file (subject to move)
    latents_file = 's3://modis-l2/SSL_MODIS_R2019_2010_latents_v2/modis_R2019_2010_latents_last_v2.h5'

    # Load em in
    basefile = os.path.basename(latents_file)
    if not os.path.isfile(basefile):
        print("Downloading latents (this is *much* faster than s3 access)...")
        ulmo_io.download_file_from_s3(basefile, latents_file)
        print("Done")
    hf = h5py.File(basefile, 'r')
    latents_train = hf['modis_latents_v2_train'][:]
    latents_valid = hf['modis_latents_v2_valid'][:]
    print("Latents loaded")

        # Table (useful for valid only)
    modis_tbl = ulmo_io.load_main_table('s3://modis-l2/Tables/MODIS_L2_std.parquet')

    # Valid
    valid = modis_tbl.pp_type == 0
    y2010 = modis_tbl.pp_file == 's3://modis-l2/PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'
    valid_tbl = modis_tbl[valid & y2010].copy()
    
    # Check
    assert latents_valid.shape[0] == len(valid_tbl)

    # Stack em
    latents = np.concatenate([latents_train, latents_valid])
    ssl_analysis.latents_umap(
        latents, np.arange(latents_train.shape[0]), 
        latents_train.shape[0]+np.arange(latents_valid.shape[0]),
        valid_tbl, fig_root='MODIS_2010_v2', debug=False,
        write_to_file='s3://modis-l2/Tables/MODIS_2010_valid_SSLv2.parquet',
        cut_prefix='modis_')


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

def model_latents_extract(opt, modis_data, model_path, 
                          save_path, save_key):
    """
    This function is used to obtain the latents of the training data.

    Args:
        opt: (Parameters) parameters used to create the model.
        modis_data: (numpy.array) modis_data used in the latents
            extraction process.
        model_path: (string) path of the model file. 
        save_path: (string) path to save the extracted latents
        save_key: (string) key of the h5py file for the latents.
    """
    # Init model
    model, _ = set_model(opt, cuda_use=opt.cuda_use)
    # Download
    print(f"Using model {model_path} for evaluation")
    model_file = os.path.basename(model_path)
    ulmo_io.download_file_from_s3(model_file, model_path)
    # Load
    print(f"Loading model")
    model_dict = torch.load(model_file)
    model.load_state_dict(model_dict['model'])
    os.remove(model_file)

    #modis_data = np.repeat(modis_data, 3, axis=1)
    num_samples = modis_data.shape[0]
    batch_size = opt.batch_size
    num_steps = num_samples // batch_size
    remainder = num_samples % batch_size
    latents_df = pd.DataFrame()

    # Process
    print(f"Processing..")
    with torch.no_grad():
        for i in trange(num_steps):
            image_batch = modis_data[i*batch_size: (i+1)*batch_size]
            import pdb; pdb.set_trace()
            image_tensor = torch.tensor(image_batch)
            if opt.cuda_use and torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            latents_tensor = model(image_tensor)
            latents_numpy = latents_tensor.cpu().numpy()
            latents_df = pd.concat([latents_df, pd.DataFrame(latents_numpy)], ignore_index=True)
        if remainder:
            image_remainder = modis_data[-remainder:]
            image_tensor = torch.tensor(image_remainder)
            if opt.cuda_use and torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            latents_tensor = model(image_tensor)
            latents_numpy = latents_tensor.cpu().numpy()
            latents_df = pd.concat([latents_df, pd.DataFrame(latents_numpy)], ignore_index=True)
            latents_numpy = latents_df.values

    # Write locally
    with h5py.File(save_path, 'a') as file:
        file.create_dataset(save_key, data=latents_numpy)
        
def main_evaluate(opt_path, model_file, 
                  preproc='_std', debug=False):
    """
    This function is used to obtain the latents of the trained models
    for all of MODIS

    Args:
        opt_path: (str) option file path.
        model: (str) Baseame of the model (in s3_outdir)
        preproc: (str, optional)
    """
    opt = option_preprocess(Params(opt_path))
    
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
                dataset_train = file['train'][:]
            else:
                dataset_train = None
            dataset_valid = file['valid'][:]
        print("Reading data is done.")

        # Remove
        os.remove(data_file)
        print(f'{data_file} removed')
    
    
        # Setup
        model_path = os.path.join(opt.s3_outdir, model_file)
        latents_file = data_file.replace(preproc, '_latents')
        latents_path = os.path.join(opt.latents_folder, latents_file) 
        if dataset_train is not None:
            print("Starting train evaluation")
            model_latents_extract(opt, dataset_train, 
                                  model_path, latents_file, key_train)
            print("Extraction of Latents of train set is done.")
        print("Starting valid evaluation")
        model_latents_extract(opt, dataset_valid, model_path, 
                              latents_file, key_valid)
        print("Extraction of Latents of valid set is done.")

        # Push to s3
        print("Uploading to s3..")
        ulmo_io.upload_file_to_s3(latents_file, latents_path)
        
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
        main_evaluate(args.opt_path, 'last.pth',
                      debug=args.debug)
        print("Evaluation Ends.")

    # run the umap
    if args.func_flag == 'umap':
        print("UMAP Starts.")
        ssl_v2_umap()
        print("UMAP Ends.")
