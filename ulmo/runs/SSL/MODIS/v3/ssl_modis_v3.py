""" SSL Analayis of MODIS -- 95% and CF 
See v3 for 96% clear [final choice]
"""
import os
from re import A
from typing import IO
import numpy as np
import pickle

import time
import h5py
import numpy as np
from tqdm.auto import trange
import argparse


import h5py
import umap

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils
from ulmo.scripts import collect_images

from ulmo.ssl import analysis as ssl_analysis
from ulmo.ssl.util import adjust_learning_rate
from ulmo.ssl.util import set_optimizer, save_model
from ulmo.ssl import latents_extraction
from ulmo.ssl import defs as ssl_defs

from ulmo.ssl.train_util import option_preprocess
from ulmo.ssl.train_util import modis_loader, set_model
from ulmo.ssl.train_util import train_model

from IPython import embed


def ssl_v3_umap(opt_path:str, debug=False, ntrain = 150000,
                umap_file:str=None, ukeys:str=None,
                outfile:str=None, ndim:int=2):
    """Run a UMAP analysis on all the MODIS L2 data

    Either 2 or 3 dimensions

    Args:
        model_name: (str) model name 
        ntrain (int, optional): Number of random latent vectors to use to train the UMAP model
        debug (bool, optional): For testing and debuggin 
        ndim (int, optional): Number of dimensions for the embedding
    """
    raise NotImplementedError("Am using umap_subset in Analysis/py/ssl_paper_analy.py")
    # Load up the options file
    opt = option_preprocess(ulmo_io.Params(opt_path))
    model_file = os.path.join(opt.model_folder, 'last.pth')

    # Load table
    modis_tbl = ulmo_io.load_main_table(opt.tbl_file)
    if ndim == 2:
        if ukeys is None:
            ukeys = ('U0', 'U1')
        for key in ukeys:
            modis_tbl[key] = 0.
    elif ndim == 3:
        modis_tbl['U3_0'] = 0.
        modis_tbl['U3_1'] = 0.
        modis_tbl['U3_2'] = 0.
    else:
        raise IOError(f"Not ready for ndim={ndim}")


    # Prep latent_files
    latents_path = os.path.join(opt.s3_outdir, opt.latents_folder)
    latent_files = ulmo_io.list_of_bucket_files(latents_path)
        #'modis-l2', 
        #prefix='SSL/latents/MODIS_R2019_2010/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm/')
                                                #prefix='SSL/SSL_v2_2012/latents/')
    latent_files = ['s3://modis-l2/'+item for item in latent_files]

    # Train the UMAP
    # Split
    y2010 = modis_tbl.pp_file == 's3://modis-l2/PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'

    # Latents file (subject to move)
    #latents_train_file = 's3://modis-l2/SSL/SSL_v2_2012/latents/MODIS_R2019_2010_95clear_128x128_latents_std.h5'
    if opt.model_root == 'MODIS_R2019_2010':
        latents_train_file = 's3://modis-l2/SSL/latents/MODIS_R2019_2010/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm/MODIS_R2019_2010_95clear_128x128_latents_std.h5'
        train = modis_tbl.pp_type == 1
        valid = modis_tbl.pp_type == 0
        cut_prefix = 'modis_'
    elif opt.model_root == 'MODIS_R2019_CF':
        valid = modis_tbl.ulmo_pp_type == 0
        train = modis_tbl.ulmo_pp_type == 1
        latents_train_file = latent_files[-10]
        assert 'R2019_2010' in latents_train_file
        cut_prefix = 'ulmo_'
    else:
        raise ValueError("Not ready for this model!!")

    # Load em in
    basefile = os.path.basename(latents_train_file)
    if not os.path.isfile(basefile):
        print("Downloading latents (this is *much* faster than s3 access)...")
        ulmo_io.download_file_from_s3(basefile, latents_train_file)
        print("Done")
    hf = h5py.File(basefile, 'r')
    print("Latents loaded")

    # Build up the training set for UMAP
    if opt.model_root == 'MODIS_R2019_2010':
        valid_tbl = modis_tbl[valid & y2010].copy()
        nvalid = len(valid_tbl)
        latents_valid = hf['valid'][:]
        # Check
        assert latents_valid.shape[0] == nvalid

        # Cut down
        train_idx = np.arange(nvalid)
        np.random.shuffle(train_idx)
        train_idx = train_idx[0:ntrain]
        latents_train = latents_valid[train_idx]
    elif opt.model_root == 'MODIS_R2019_CF':
        latents_valid = hf['valid'][:]
        latents_train = hf['train'][:]
        # Grab CF
        valid_cf_idx = modis_tbl[y2010 & valid].pp_idx.values
        train_cf_idx = modis_tbl[y2010 & train].pp_idx.values
        latents_valid_cf = latents_valid[valid_cf_idx,...]
        latents_train_cf = latents_train[train_cf_idx,...]
        # Merge
        latents_train = np.concatenate([latents_valid_cf, latents_train_cf])
        ntrain = latents_train.shape[0]
    
    # Train the UMAP
    if umap_file is not None:
        latents_mapping = pickle.load(ulmo_io.open(umap_file, "rb")) 
    else:
        print(f"Running UMAP on a {ntrain} subset of {basefile}..")
        reducer_umap = umap.UMAP(n_components=ndim)
        latents_mapping = reducer_umap.fit(latents_train)
        print("Done..")

    # Loop on em all
    if debug:
        latent_files = latent_files[0:1]

    # Evaluate with the UMAP
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
        pp_idx = modis_tbl[valid_idx].pp_idx.values
        if ndim == 2:
            modis_tbl.loc[valid_idx, ukeys[0]] = valid_embedding[pp_idx,0]
            modis_tbl.loc[valid_idx, ukeys[1]] = valid_embedding[pp_idx,1]
        elif ndim == 3:
            modis_tbl.loc[valid_idx, 'U3_0'] = valid_embedding[pp_idx,0]
            modis_tbl.loc[valid_idx, 'U3_1'] = valid_embedding[pp_idx,1]
            modis_tbl.loc[valid_idx, 'U3_2'] = valid_embedding[pp_idx,2]
        
        # Train?
        train_idx = train & yidx
        if 'train' in hf.keys() and (np.sum(train_idx) > 0):
            pp_idx = modis_tbl[train_idx].pp_idx.values
            if ndim == 2:
                modis_tbl.loc[train_idx, ukeys[0]] = train_embedding[pp_idx,0]
                modis_tbl.loc[train_idx, ukeys[1]] = train_embedding[pp_idx,1]
            elif ndim == 3:
                modis_tbl.loc[train_idx, 'U3_0'] = train_embedding[pp_idx,0]
                modis_tbl.loc[train_idx, 'U3_1'] = train_embedding[pp_idx,1]
                modis_tbl.loc[train_idx, 'U3_2'] = train_embedding[pp_idx,2]


        hf.close()

        # Clean up
        print(f"Done with {basefile}.  Cleaning up")
        os.remove(basefile)

    # Vet
    assert cat_utils.vet_main_table(modis_tbl, cut_prefix=cut_prefix)

    # Final write
    if not debug:
        if outfile is None:
            outfile = opt.tbl_file
        ulmo_io.write_main_table(modis_tbl, outfile)
    else:
        embed(header='205 of ssl')
        
def main_train(opt_path: str, debug=False, restore=False, save_file=None):
    """Train the model

    Previously ---
    After running on 2012 without a validation dataset,
    I have now switched to running on 2010.  And to confuse
    everyone, I am going to use the valid set for training
    and the train set for validation.  This is to have ~100,000
    for validation and ~800,000 for training.  
    But note it is done in the opt_path file.
    Yup, that is confusing

    Now -- (June 2022)
    We build a single file satisfying the clear_fraction criterion
    and train/valid are properly named. 

    Args:
        opt_path (str): Path + filename of options file
        debug (bool): 
        restore (bool):
        save_file (str): 
    """
    # loading parameters json file
    opt = ulmo_io.Params(opt_path)
    if debug:
        opt.epochs = 2
    opt = option_preprocess(opt)

    # Vet
    #assert cat_utils.vet_main_table(opt.__dict__, 
    #                                data_model=ssl_defs.ssl_opt_dmodel)

    # Save opts                                    
    opt.save(os.path.join(opt.model_folder, 
                          os.path.basename(opt_path)))
    
    # build model and criterion
    model, criterion = set_model(opt, cuda_use=opt.cuda_use)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    loss_train, loss_step_train, loss_avg_train = [], [], []
    loss_valid, loss_step_valid, loss_avg_valid = [], [], []

    for epoch in trange(1, opt.epochs + 1): 
        # build data loader
        # NOTE: For 2010 we are swapping the roles of valid and train!!
        train_loader = modis_loader(opt)
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, losses_step, losses_avg = train_model(
            train_loader, model, criterion, optimizer, epoch, opt, 
            cuda_use=opt.cuda_use)

        # record train loss
        loss_train.append(loss)
        loss_step_train += losses_step
        loss_avg_train += losses_avg

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # Free up memory
        del train_loader

        # Validate?
        if epoch % opt.valid_freq == 0:
            # Data Loader
            valid_loader = modis_loader(opt, valid=True)
            #
            epoch_valid = epoch // opt.valid_freq
            time1_valid = time.time()
            loss, losses_step, losses_avg = train_model(
                valid_loader, model, criterion, optimizer, epoch_valid, opt, 
                cuda_use=opt.cuda_use, update_model=False)
           
            # record valid loss
            loss_valid.append(loss)
            loss_step_valid += losses_step
            loss_avg_valid += losses_avg
        
            time2_valid = time.time()
            print('valid epoch {}, total time {:.2f}'.format(epoch_valid, time2_valid - time1_valid))

            # Free up memory
            del valid_loader 

        # Save model?
        if (epoch % opt.save_freq) == 0:
            # Save locally
            save_file = os.path.join(opt.model_folder,
                                     f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)
            
    # save the last model local
    save_file = os.path.join(opt.model_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # Save the losses
    if not os.path.isdir(f'{opt.model_folder}/learning_curve/'):
        os.mkdir(f'{opt.model_folder}/learning_curve/')
        
    losses_file_train = os.path.join(opt.model_folder,'learning_curve',
                                     f'{opt.model_name}_losses_train.h5')
    losses_file_valid = os.path.join(opt.model_folder,'learning_curve',
                                     f'{opt.model_name}_losses_valid.h5')
    
    with h5py.File(losses_file_train, 'w') as f:
        f.create_dataset('loss_train', data=np.array(loss_train))
        f.create_dataset('loss_step_train', data=np.array(loss_step_train))
        f.create_dataset('loss_avg_train', data=np.array(loss_avg_train))
    with h5py.File(losses_file_valid, 'w') as f:
        f.create_dataset('loss_valid', data=np.array(loss_valid))
        f.create_dataset('loss_step_valid', data=np.array(loss_step_valid))
        f.create_dataset('loss_avg_valid', data=np.array(loss_avg_valid))
        

def main_evaluate(opt_path, model_name, 
                  preproc='_std', debug=False, 
                  clobber=False):
    """
    This function is used to obtain the latents of the trained models
    for all of MODIS

    Args:
        opt_path: (str) option file path.
        model_name: (str) model name 
        preproc: (str, optional)
            Type of pre-processing
        clobber: (bool, optional)
            If true, over-write any existing file
    """
    # Parse the model
    opt = option_preprocess(ulmo_io.Params(opt_path))
    model_file = os.path.join(opt.s3_outdir,
        opt.model_folder, 'last.pth')

    # Load up the table
    print(f"Grabbing table: {opt.tbl_file}")
    modis_tbl = ulmo_io.load_main_table(opt.tbl_file)

    # Grab the model
    print(f"Grabbing model: {model_file}")
    model_base = os.path.basename(model_file)
    ulmo_io.download_file_from_s3(model_base, model_file)
    
    # Data files
    all_pp_files = ulmo_io.list_of_bucket_files('modis-l2', 'PreProc')
    pp_files = []
    for ifile in all_pp_files:
        if preproc in ifile:
            pp_files.append(ifile)

    # Loop on files
    if debug:
        pp_files = pp_files[0:1]

    latents_path = os.path.join(opt.s3_outdir, opt.latents_folder)
    # Grab existing for clobber
    if not clobber:
        parse_s3 = ulmo_io.urlparse(opt.s3_outdir)
        existing_files = [os.path.basename(ifile) for ifile in ulmo_io.list_of_bucket_files('modis-l2',
                                                      prefix=os.path.join(parse_s3.path[1:],
                                                                        opt.latents_folder))
                          ]
    else:
        existing_files = []

    for ifile in pp_files:
        print(f"Working on {ifile}")
        data_file = os.path.basename(ifile)

        # Setup
        latents_file = data_file.replace('_preproc', '_latents')
        if latents_file in existing_files and not clobber:
            print(f"Not clobbering {latents_file} in s3")
            continue
        s3_file = os.path.join(latents_path, latents_file) 

        # Download
        s3_preproc_file = f's3://modis-l2/PreProc/{data_file}'
        if not os.path.isfile(data_file):
            ulmo_io.download_file_from_s3(data_file, s3_preproc_file)

        # Ready to write
        latents_hf = h5py.File(latents_file, 'w')

        # Read
        with h5py.File(data_file, 'r') as file:
            if 'train' in file.keys():
                train=True
            else:
                train=False

        # Train?
        if train: 
            print("Starting train evaluation")
            latents_numpy = latents_extraction.model_latents_extract(
                opt, data_file, 'train', model_base, None, None)
            latents_hf.create_dataset('train', data=latents_numpy)
            print("Extraction of Latents of train set is done.")

        # Valid
        print("Starting valid evaluation")
        latents_numpy = latents_extraction.model_latents_extract(
            opt, data_file, 'valid', model_base, None, None)
        latents_hf.create_dataset('valid', data=latents_numpy)
        print("Extraction of Latents of valid set is done.")

        # Close
        latents_hf.close()

        # Push to s3
        print("Uploading to s3..")
        ulmo_io.upload_file_to_s3(latents_file, s3_file)

        # Remove data file
        if not debug:
            os.remove(data_file)
            print(f'{data_file} removed')

def sub_tbl_2010():

    # Load table
    tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Split
    valid = modis_tbl.pp_type == 0
    y2010 = modis_tbl.pp_file == 's3://modis-l2/PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'
    valid_tbl = modis_tbl[valid & y2010].copy()

    # Write
    ulmo_io.write_main_table(valid_tbl, 'MODIS_2010_valid_SSLv2.parquet', to_s3=False)
    

def prep_cloud_free(clear_fraction=96, local=True, 
                    img_shape=(64,64), debug=False, 
                    outfile='MODIS_SSL_96clear_images.h5',
                    new_tbl_file='s3://modis-l2/Tables/MODIS_SSL_96clear.parquet'): 
    """ Generate a data file for SSL traiing on a subset of 
    MODIS L2 that are "cloud free"  (>= 96% clear)

    Args:
        clear_fraction (float, optional): [description]. Defaults to 96
        local (bool, optional): [description]. Defaults to True.
        img_shape (tuple, optional): [description]. Defaults to (64,64).
        debug (bool, optional): [description]. Defaults to False.
        outfile (str, optional): [description]. Defaults to 'MODIS_SSL_cloud_free_images.h5'.
    """

    # Load table
    if local:
        tbl_file = os.path.join(os.getenv('SST_OOD'), 'MODIS_L2', 'Tables', 
                            'MODIS_L2_std.parquet')
    else:
        tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    print("Loading the table..")
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Restrict to cloud free
    cloud_free = modis_tbl.clear_fraction < (1-clear_fraction/100)
    cfree_tbl = modis_tbl[cloud_free].copy()
    print(f"We have {len(cfree_tbl)} images satisfying the clear_fraction={clear_fraction} criterion")

    # Save Ulmo pp_type
    cfree_tbl['ulmo_pp_type'] = cfree_tbl.pp_type.values.copy()

    # Keep it simple and avoid 2010 train images
    all_ulmo_valid = cfree_tbl.ulmo_pp_type == 0
    all_ulmo_valid_idx = np.where(all_ulmo_valid)[0]
    nulmo_valid = np.sum(all_ulmo_valid)

    # Choose 600,000 random for train and 150,000 for valid
    nSSL_train = 600000
    nSSL_valid = 150000

    # Prepare
    train_imgs = np.zeros((nSSL_train, img_shape[0], img_shape[1])).astype(np.float32)
    valid_imgs = np.zeros((nSSL_valid, img_shape[0], img_shape[1])).astype(np.float32)

    indices = np.random.choice(np.arange(nulmo_valid),
                               size=nSSL_train+nSSL_valid,
                               replace=False)
    train = indices[0:nSSL_train]
    valid = indices[nSSL_train:]

    # Set cfree pp_type (for SSL)
    pp_types = np.ones(len(cfree_tbl)).astype(int)*-1
    pp_types[train] = 1
    pp_types[valid] = 0
    cfree_tbl.pp_type = pp_types

    # This needs to be a copy
    img_tbl = cfree_tbl[all_ulmo_valid].copy()
    train_img_pp = img_tbl.pp_type == 1
    valid_img_pp = img_tbl.pp_type == 0
    
    # Loop on PreProc files
    print("Building the file for SSL training and validation")
    pp_files = np.unique(img_tbl.pp_file)
    ivalid, itrain = 0, 0
    for pp_file in pp_files:
        ipp = img_tbl.pp_file == pp_file
        # Local?
        if local:
            dpath = os.path.join(os.getenv('SST_OOD'), 'MODIS_L2', 'PreProc')
            ifile = os.path.join(dpath, os.path.basename(pp_file))
        else:
            embed(header='Not setup for this')
        # Open
        print(f"Working on: {ifile}")
        hf = h5py.File(ifile, 'r')
        all_ulmo_valid = hf['valid'][:]

        # Valid (Ulmo)
        if np.any(valid_img_pp & ipp):
            # Fastest to grab em all
            iidx = np.where(valid_img_pp & ipp)[0]
            idx = img_tbl.pp_idx.values[iidx]
            n_new = len(idx)
            valid_imgs[ivalid:ivalid+n_new, ...] = all_ulmo_valid[idx, 0, :, :]
            ivalid += n_new

        # Train (Ulmo)
        if np.any(train_img_pp & ipp):
            # Fastest to grab em all
            iidx = np.where(train_img_pp & ipp)[0]
            idx = img_tbl.pp_idx.values[iidx]
            n_new = len(idx)
            train_imgs[itrain:itrain+n_new, ...] = all_ulmo_valid[idx, 0, :, :]
            itrain += n_new

        del all_ulmo_valid

        hf.close()
        # 
        if debug:
            break

    # Write
    out_h5 = h5py.File(outfile, 'w')
    out_h5.create_dataset('train', data=train_imgs.reshape((train_imgs.shape[0],1,img_shape[0], img_shape[1]))) 
    out_h5.create_dataset('train_indices', data=all_ulmo_valid_idx[train])  # These are the cloud free indices
    out_h5.create_dataset('valid', data=valid_imgs.reshape((valid_imgs.shape[0],1, img_shape[0], img_shape[1])))
    out_h5.create_dataset('valid_indices', data=all_ulmo_valid_idx[valid])  # These are the cloud free indices
    out_h5.close()
    print(f"Wrote: {outfile}")

    # Push to s3
    print("Uploading to s3")
    ulmo_io.upload_file_to_s3(
        outfile, 's3://modis-l2/SSL/preproc/'+outfile)
    
    # Table
    assert cat_utils.vet_main_table(
        cfree_tbl, cut_prefix='ulmo_')
    ulmo_io.write_main_table(cfree_tbl, new_tbl_file)

def run_collect_images(pargs):
    if pargs.table_file is None:
        raise IOError("You must specify --table_file !")
    if pargs.outfile is None:
        raise IOError("You must specify --outfile !")
    # Use the defaults
    pargs.image_path = None
    pargs.nimages = None
    # Run it
    collect_images.main(pargs)

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("argument for training.")
    parser.add_argument("--opt_path", type=str, 
                        default='opts_96clear_ssl.json',
                        help="Path to options file. Defaults to 96percent clear")
    parser.add_argument("--func_flag", type=str, 
                        help="flag of the function to be execute: train,evaluate,umap,umap_ndim3,sub2010,collect")
    parser.add_argument("--model", type=str, 
                        default='2010', help="Short name of the model used [2010,CF]")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Clobber existing files')
    parser.add_argument("--outfile", type=str, 
                        help="Path to output file")
    parser.add_argument("--umap_file", type=str, 
                        help="Path to UMAP pickle file for analysis")
    parser.add_argument("--table_file", type=str, 
                        help="Path to Table file")
    parser.add_argument("--cf", type=float, 
                        help="Clear fraction (e.g. 96)")
    args = parser.parse_args()
    
    return args

        
if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    
    # run the 'main_train()' function.
    if args.func_flag == 'train':
        print("Training Starts.")
        main_train(args.opt_path, debug=args.debug)
        print("Training Ends.")
    
    # run the "main_evaluate()" function.
    if args.func_flag == 'evaluate':
        print("Evaluation Starts.")
        main_evaluate(args.opt_path, args.model,
                      debug=args.debug, clobber=args.clobber)
        print("Evaluation Ends.")

    # run the umap
    if args.func_flag[0:4] == 'umap':
        print("UMAP Starts.")
        if args.debug:
            print("In debug mode!!")
        if args.func_flag == 'umap_ndim3':
            ndim = 3
        else:
            ndim = 2
        if args.func_flag == 'umap_DT1':
            ukeys = ('UT1_0', 'UT1_1')
        else:
            ukeys = None
        ssl_v3_umap(args.opt_path, debug=args.debug, 
                    ndim=ndim,
                    umap_file=args.umap_file, 
                    outfile=args.outfile,
                    ukeys=ukeys)
        print("UMAP Ends.")

    # 
    if args.func_flag == 'sub2010':
        sub_tbl_2010()

    # Prep for cloud free
    if args.func_flag == 'prep_cloud_free':
        prep_cloud_free(debug=args.debug,
                        clear_fraction=args.cf,
                        outfile=args.outfile)

    # 
    if args.func_flag == 'collect':
        run_collect_images(args)


# Re-run with 96% cloud free -- June 2022

# Training images -- on profx (DONE) 2022-06-04 -- Done wrong
# on profx (DONE) 2022-06-08 
# python ssl_modis_v3.py --func_flag prep_cloud_free --cf 96 --outfile MODIS_SSL_96clear_images.h5

# EXTRACTION WAS DONE WITH THE v2 FILE
# Re-run with 96% cloud free -- June 2022
#  BROKE OFF HERE AND STARTED v3
# python ssl_modis_v2.py --func_flag prep_cloud_free


# TRAIN :: Run in cloud
#  python ./ssl_modis_v3.py --opt_path opts_96clear_ssl.json --func_flag train;

# EVAL :: Run in cloud
#  

# UMAP ::