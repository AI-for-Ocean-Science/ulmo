""" Module for Ulmo analysis on VIIRS 2013"""
import os
import numpy as np

import time
import h5py
import numpy as np
from tqdm.auto import trange
import argparse


import h5py
import umap

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
    parser.add_argument("--opt_path", type=str, 
                        default='./experiments/modis_model_v2/opts_2010.json',
                        help="Path to options file. Defaults to local + 2010")
    parser.add_argument("--func_flag", type=str, help="flag of the function to be execute: 'train' or 'evaluate' or 'umap'.")
        # JFH Should the default now be true with the new definition.
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    args = parser.parse_args()
    
    return args

def ssl_v2_umap(debug=False):
    """Run a UMAP analysis on all the MODIS L2 data

    Args:
        debug (bool, optional): [description]. Defaults to False.
        orig (bool, optional): [description]. Defaults to False.
    """
    # Load table
    tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    modis_tbl = ulmo_io.load_main_table(tbl_file)
    modis_tbl['U0'] = 0.
    modis_tbl['U1'] = 0.


    # Prep latent_files
    latent_files = ulmo_io.list_of_bucket_files('modis-l2',
                                                prefix='SSL/SSL_v2_2012/latents/')
    latent_files = ['s3://modis-l2/'+item for item in latent_files]

    # Train the UMAP
    # Split
    train = modis_tbl.pp_type == 1
    valid = modis_tbl.pp_type == 0
    y2010 = modis_tbl.pp_file == 's3://modis-l2/PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'
    valid_tbl = modis_tbl[valid & y2010].copy()
    nvalid = len(valid_tbl)

    # Latents file (subject to move)
    latents_train_file = 's3://modis-l2/SSL/SSL_v2_2012/latents/MODIS_R2019_2010_95clear_128x128_latents_std.h5'

    # Load em in
    basefile = os.path.basename(latents_train_file)
    if not os.path.isfile(basefile):
        print("Downloading latents (this is *much* faster than s3 access)...")
        ulmo_io.download_file_from_s3(basefile, latents_train_file)
        print("Done")
    hf = h5py.File(basefile, 'r')
    latents_valid = hf['valid'][:]
    print("Latents loaded")

    # Check
    assert latents_valid.shape[0] == nvalid

    ntrain = 150000
    train_idx = np.arange(nvalid)
    np.random.shuffle(train_idx)
    train_idx = train_idx[0:ntrain]
    latents_train = latents_valid[train_idx]

    print(f"Running UMAP on a {ntrain} subset of valid..")
    reducer_umap = umap.UMAP()
    latents_mapping = reducer_umap.fit(latents_train)
    print("Done..")

    # Loop on em all
    if debug:
        latent_files = latent_files[0:1]

    for latents_file in latent_files:
        basefile = os.path.basename(latents_file)
        year = int(basefile[12:16])
        # Download?
        if not os.path.isfile(basefile):
            print(f"Downloading {latents_file} (this is *much* faster than s3 access)...")
            ulmo_io.download_file_from_s3(basefile, latents_file)

        #  Load and apply
        hf = h5py.File(basefile, 'r')

        # Train
        if 'train' in hf.keys():
            print("Embedding the training..")
            latents_train = hf['train'][:]
            train_embedding = latents_mapping.transform(latents_train)

        # Valid
        print("Embedding valid..")
        latents_valid = hf['valid'][:]
        valid_embedding = latents_mapping.transform(latents_valid)

        # Save to table
        yidx = modis_tbl.pp_file == f's3://modis-l2/PreProc/MODIS_R2019_{year}_95clear_128x128_preproc_std.h5'
        valid_idx = valid & yidx
        modis_tbl.loc[valid_idx, 'U0'] = valid_embedding[:,0]
        modis_tbl.loc[valid_idx, 'U1'] = valid_embedding[:,1]
        
        # Train?
        train_idx = train & yidx
        if 'train' in hf.keys() and (np.sum(train_idx) > 0):
            modis_tbl.loc[train_idx, 'U0'] = train_embedding[:,0]
            modis_tbl.loc[train_idx, 'U1'] = train_embedding[:,1]


        hf.close()

        # Clean up
        print(f"Done with {basefile}.  Cleaning up")
        os.remove(basefile)

    # Vet
    assert cat_utils.vet_main_table(modis_tbl, cut_prefix='modis_')

    # Final write
    if not debug:
        ulmo_io.write_main_table(modis_tbl, tbl_file) 


def main_train(opt_path: str):
    """Train the model

    After running on 2012 without a validation dataset,
    I have now switched to running on 2010.  And to confuse
    everyone, I am going to use the valid set for training
    and the train set for validation.  This is to have ~100,000
    for validation and ~800,000 for training.  

    Yup, that is confusing

    Args:
        opt_path (str): Path + filename of options file
    """
    # loading parameters json file
    opt = Params(opt_path)
    opt = option_preprocess(opt)

    # build data loaders -- 
    # NOTE: For 2010 we are swapping the roles of valid and train!!
    train_loader = modis_loader(opt)
    valid_loader = modis_loader(opt, valid=True)

    # build model and criterion
    model, criterion = set_model(opt, cuda_use=opt.cuda_use)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    # training routine
    for epoch in trange(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, losses_step, losses_avg = train_model(
            train_loader, model, criterion, optimizer, epoch, opt, 
            cuda_use=opt.cuda_use)
        #loss = train_model(train_loader, model, criterion, 
        #                   optimizer, epoch, opt, cuda_use=opt.cuda_use)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # Validate?
        if epoch % opt.valid_freq == 0:
            epoch_valid = epoch // opt.valid_freq
            time1_valid = time.time()
            loss, losses_step, losses_avg = train_model(
                valid_loader, model, criterion, epoch_valid, opt, 
                cuda_use=opt.cuda_use, update_model=False)
           
            # record valid loss
            loss_valid.append(loss)
            loss_step_valid += losses_step
            loss_avg_valid += losses_avg
        
            time2_valid = time.time()
            print('valid epoch {}, total time {:.2f}'.format(epoch_valid, time2_valid - time1_valid))

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

    # Save the losses
    if not os.path.isdir('./learning_curve/'):
        os.mkdir('./learning_curve/')
        
    losses_file_train = f'./learning_curve/{opt.dataset}_losses_train.h5'
    losses_file_valid = f'./learning_curve/{opt.dataset}_losses_valid.h5'
    
    with h5py.File(losses_file_train, 'w') as f:
        f.create_dataset('loss_train', data=np.array(loss_train))
        f.create_dataset('loss_step_train', data=np.array(loss_step_train))
        f.create_dataset('loss_avg_train', data=np.array(loss_avg_train))
    with h5py.File(losses_file_valid, 'w') as f:
        f.create_dataset('loss_valid', data=np.array(loss_valid))
        f.create_dataset('loss_step_valid', data=np.array(loss_step_valid))
        f.create_dataset('loss_avg_valid', data=np.array(loss_avg_valid))

        
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
        print(f"Working on {ifile}")
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
        latents_hf = h5py.File(latents_file, 'w')

        # Train?
        if train: 
            print("Starting train evaluation")
            latents_numpy = latents_extraction.model_latents_extract(opt, data_file, 
                'train', model_base, None, None)
            latents_hf.create_dataset('train', data=latents_numpy)
            print("Extraction of Latents of train set is done.")

        # Valid
        print("Starting valid evaluation")
        latents_numpy = latents_extraction.model_latents_extract(opt, data_file, 
                'valid', model_base, None, None)
        latents_hf.create_dataset('valid', data=latents_numpy)
        print("Extraction of Latents of valid set is done.")

        # Close
        latents_hf.close()

        # Push to s3
        print("Uploading to s3..")
        ulmo_io.upload_file_to_s3(latents_file, latents_path)

        # Remove data file
        if not debug:
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
        if args.debug:
            print("In debug mode!!")
        ssl_v2_umap(debug=args.debug)
        print("UMAP Ends.")
