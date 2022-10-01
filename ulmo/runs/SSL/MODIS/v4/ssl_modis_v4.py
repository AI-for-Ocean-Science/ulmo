""" SSL Analayis of MODIS -- 
96% clear 
New set of Augmentations
"""
from genericpath import isfile
import os
from typing import IO
import numpy as np

import time
import h5py
import numpy as np
from tqdm.auto import trange
import argparse

import pandas
from sklearn.utils import shuffle
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import h5py

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

from ulmo.preproc import io as pp_io 
from ulmo.modis import utils as modis_utils
from ulmo.modis import extract as modis_extract

from IPython import embed


def main_train(opt_path: str, debug=False, restore=False, save_file=None):
    """Train the model

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


#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def extract_modis(debug=False, n_cores=20, 
                       nsub_files=1000,
                       ndebug_files=100):
    """Extract "cloud free" images for 2020 and 2021

    Args:
        debug (bool, optional): [description]. Defaults to False.
        n_cores (int, optional): Number of cores to use. Defaults to 20.
        nsub_files (int, optional): Number of sub files to process at a time. Defaults to 5000.
        ndebug_files (int, optional): [description]. Defaults to 0.
    """
    # 10 cores took 6hrs
    # 20 cores took 3hrs

    if debug:
        tbl_file = 's3://modis-l2/Tables/MODIS_L2_std_debug.parquet'
    else:
        tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    # Pre-processing (and extraction) settings
    pdict = pp_io.load_options('standard')
    
    # 2013 
    print("Grabbing the file list")
    all_modis_files = ulmo_io.list_of_bucket_files('modis-l2')
    files = []
    bucket = 's3://modis-l2/'
    for ifile in all_modis_files:
        if ('data/2020' in ifile) or ('data/2021' in ifile):
            files.append(bucket+ifile)

    # Output
    if debug:
        save_path = ('MODIS_2019'
                 '_{}clear_{}x{}_tst_inpaint.h5'.format(pdict['clear_threshold'],
                                                    pdict['field_size'],
                                                    pdict['field_size']))
    else:                                                
        save_path = ('MODIS_R2019'
                 '_{}clear_{}x{}_inpaint.h5'.format(pdict['clear_threshold'],
                                                    pdict['field_size'],
                                                    pdict['field_size']))
    s3_filename = 's3://modis-l2/Extractions/{}'.format(save_path)

    if debug:
        # Grab 100 random
        #files = shuffle(files, random_state=1234)
        files = files[:ndebug_files]  # 10%
        n_cores = 4
        #files = files[:100]

    # Setup for preproc
    map_fn = partial(modis_extract.extract_file, '',
                     field_size=(pdict['field_size'], pdict['field_size']),
                     CC_max=1.-pdict['clear_threshold'] / 100.,
                     nadir_offset=pdict['nadir_offset'],
                     temp_bounds=tuple(pdict['temp_bounds']),
                     nrepeat=pdict['nrepeat'],
                     inpaint=True)

    # Local file for writing
    f_h5 = h5py.File(save_path, 'w')
    print("Opened local file: {}".format(save_path))
    
    nloop = len(files) // nsub_files + ((len(files) % nsub_files) > 0)
    metadata = None
    #if debug:
    #    embed(header='464 of v4')
    for kk in range(nloop):
        # Zero out
        fields, inpainted_masks = None, None
        #
        i0 = kk*nsub_files
        i1 = min((kk+1)*nsub_files, len(files))
        print('Files: {}:{} of {}'.format(i0, i1, len(files)))
        sub_files = files[i0:i1]

        # Download
        basefiles = []
        print("Downloading files from s3...")
        for ifile in sub_files:
            basename = os.path.basename(ifile)
            basefiles.append(basename)
            # Already here?
            if os.path.isfile(basename):
                continue
            ulmo_io.download_file_from_s3(basename, ifile)
        print("All Done!")

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(sub_files) // n_cores if len(sub_files) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, sub_files,
                                             chunksize=chunksize), 
                                total=len(sub_files)))

        # Trim None's
        answers = [f for f in answers if f is not None]
        try:
            fields = np.concatenate([item[0] for item in answers])
        except:
            import pdb; pdb.set_trace()
        
        inpainted_masks = np.concatenate([item[1] for item in answers])
        if metadata is None:
            metadata = np.concatenate([item[2] for item in answers])
        else:
            metadata = np.concatenate([metadata]+[item[2] for item in answers], axis=0)
        del answers

        # Write
        if kk == 0:
            f_h5.create_dataset('fields', data=fields, 
                                compression="gzip", chunks=True,
                                maxshape=(None, fields.shape[1], fields.shape[2]))
            f_h5.create_dataset('inpainted_masks', data=inpainted_masks,
                                compression="gzip", chunks=True,
                                maxshape=(None, inpainted_masks.shape[1], inpainted_masks.shape[2]))
        else:
            # Resize
            for key in ['fields', 'inpainted_masks']:
                f_h5[key].resize((f_h5[key].shape[0] + fields.shape[0]), axis=0)
            # Fill
            f_h5['fields'][-fields.shape[0]:] = fields
            f_h5['inpainted_masks'][-fields.shape[0]:] = inpainted_masks
    
        # Remove em
        for ifile in sub_files:
            basename = os.path.basename(ifile)
            os.remove(basename)

    # Metadata
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 
               'clear_fraction']
    dset = f_h5.create_dataset('metadata', data=metadata.astype('S'))
    dset.attrs['columns'] = columns
    # Close
    f_h5.close() 

    # Table time
    modis_table = pandas.DataFrame()
    modis_table['filename'] = [item[0] for item in metadata]
    modis_table['row'] = [int(item[1]) for item in metadata]
    modis_table['col'] = [int(item[2]) for item in metadata]
    modis_table['lat'] = [float(item[3]) for item in metadata]
    modis_table['lon'] = [float(item[4]) for item in metadata]
    modis_table['clear_fraction'] = [float(item[5]) for item in metadata]
    modis_table['field_size'] = pdict['field_size']
    basefiles = [os.path.basename(ifile) for ifile in modis_table.filename.values]
    modis_table['datetime'] = modis_utils.times_from_filenames(basefiles, ioff=-1, toff=0)
    modis_table['ex_filename'] = s3_filename

    # Vet
    assert cat_utils.vet_main_table(modis_table)

    # Final write
    ulmo_io.write_main_table(modis_table, tbl_file)
    
    # Push to s3
    print("Pushing to s3")
    ulmo_io.upload_file_to_s3(save_path, s3_filename)
    #print("Run this:  s3 put {} s3://modis-l2/Extractions/{}".format(
    #    save_path, save_path))
    #process = subprocess.run(['s4cmd', '--force', '--endpoint-url',
    #    'https://s3.nautilus.optiputer.net', 'put', save_path, 
    #    s3_filename])

def calc_dt40(debug=False):
    pass

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("argument for training.")
    parser.add_argument("--opt_path", type=str, 
                        default='opts_ssl_modis_v4.json',
                        help="Path to options file")
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

    # python ssl_modis_v4.py --func_flag DT40 --debug
    if args.func_flag == 'DT40':
        calc_dt40(debug=args.debug)

    # python ssl_modis_v4.py --func_flag extract_new --debug
    if args.func_flag == 'extract_new':
        extract_modis(debug=args.debug)
    