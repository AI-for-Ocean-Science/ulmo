""" SSL Analayis of MODIS -- 
96% clear 
New set of Augmentations
"""
import os
from typing import IO
import numpy as np

import time
import h5py
import numpy as np
from tqdm.auto import trange
import argparse

import pandas
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import h5py

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils
from ulmo.preproc import utils as pp_utils

from ulmo.ssl.util import adjust_learning_rate
from ulmo.ssl.util import set_optimizer, save_model
from ulmo.ssl import latents_extraction

from ulmo.ssl.train_util import option_preprocess
from ulmo.ssl.train_util import modis_loader, set_model
from ulmo.ssl.train_util import train_model

from ulmo.preproc import io as pp_io 
from ulmo.modis import utils as modis_utils
from ulmo.modis import extract as modis_extract
from ulmo.analysis import evaluate as ulmo_evaluate 

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
        

def main_ssl_evaluate(opt_path, preproc='_std', debug=False, 
                  clobber=False):
    """
    This function is used to obtain the SSL latents of the trained models
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


#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def extract_modis(debug=False, n_cores=10, local=False, 
                       nsub_files=1000,
                       ndebug_files=100,
                       intermediate_s3=False, years=[2020, 2021],
                       use_prev=False):
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
        tbl_file = 's3://modis-l2/Tables/MODIS_L2_20202021_debug.parquet'
    else:
        tbl_file = 's3://modis-l2/Tables/MODIS_L2_20202021.parquet'
    # Pre-processing (and extraction) settings
    pdict = pp_io.load_options('standard')

    # Setup for preproc
    map_fn = partial(modis_extract.extract_file,
                     field_size=(pdict['field_size'], pdict['field_size']),
                     CC_max=1.-pdict['clear_threshold'] / 100.,
                     nadir_offset=pdict['nadir_offset'],
                     temp_bounds=tuple(pdict['temp_bounds']),
                     nrepeat=pdict['nrepeat'],
                     inpaint=True)

    print("Grabbing the file list")
    if not local:
        all_modis_files = ulmo_io.list_of_bucket_files('modis-l2')

    
    modis_tables = []
    for year in years:
        files = []
        # Grab em
        if local:
            local_path = os.path.join(os.getenv('MODIS_DATA'), 'night', f'{year}') 
            for root, dirs, ifiles in os.walk(os.path.abspath(local_path)):
                for ifile in ifiles:
                    files.append(os.path.join(root,ifile))
        else:
            bucket = 's3://modis-l2/'
            for ifile in all_modis_files:
                if ('data/'+str(year) in ifile): 
                    if ifile.endswith('.nc'):
                        files.append(bucket+ifile)

        nfiles = len(files)
        print(f'We have {nfiles} files for {year}')

        # Output
        if debug:
            save_path = ('MODIS_R2019'
                    '_{}_{}clear_{}x{}_tst_inpaint.h5'.format(
                        year,
                        pdict['clear_threshold'], 
                        pdict['field_size'], 
                        pdict['field_size']))
        else:                                                
            save_path = ('MODIS_R2019'
                    '_{}_{}clear_{}x{}_inpaint.h5'.format(
                        year,
                        pdict['clear_threshold'],
                        pdict['field_size'],
                        pdict['field_size']))
        s3_filename = 's3://modis-l2/Extractions/{}'.format(save_path)

        if debug:
            # Grab 100 random
            #files = shuffle(files, random_state=1234)
            files = files[:ndebug_files]  # 10%
            n_cores = 4
            #files = files[:100]

        # Load previous?
        if use_prev and ulmo_io.urlparse(s3_filename).path[1:] in all_modis_files:
            # Download
            prev_file = save_path.replace('inpaint', 'inpaint_prev')
            ulmo_io.download_file_from_s3(prev_file, s3_filename, 
                                          clobber_local=True)
            # Load
            f_prev = h5py.File(prev_file, 'r')
            print(f"Using: {prev_file}")
            # Load CSV file
            s3_csv_filename = 's3://modis-l2/TMP/curr_metadata_{}.csv'.format(year)
            csv_filename = 'curr_metadata_{}.csv'.format(year)
            ulmo_io.download_file_from_s3(csv_filename, s3_csv_filename, clobber_local=True)
            prev_meta = pandas.read_csv('curr_metadata_{}.csv'.format(year))
            embed(header='332 of v4')
        else:
            f_prev = None

        # Local file for writing
        f_h5 = h5py.File(save_path, 'w')
        print("Opened local file: {}".format(save_path))
        
        nloop = len(files) // nsub_files + ((len(files) % nsub_files) > 0)
        metadata = None
        if debug:
            embed(header='464 of v4')
        bad_files = []
        for kk in range(nloop):
            if f_prev is not None:
                if kk <= int(prev_meta['kk'].max()):
                    print(f"Skipping kk={kk}")
                    continue
            # Zero out
            fields, inpainted_masks = None, None
            #
            i0 = kk*nsub_files
            i1 = min((kk+1)*nsub_files, len(files))
            print('Files: {}:{} of {}'.format(i0, i1, len(files)))
            sub_files = files[i0:i1]

            # Download
            basefiles = []
            if not local:
                print("Downloading files from s3...")
            for ifile in sub_files:
                if local:
                    basefiles = sub_files
                else:
                    basename = os.path.basename(ifile)
                    basefiles.append(basename)
                    # Already here?
                    if os.path.isfile(basename):
                        continue
                    try:
                        ulmo_io.download_file_from_s3(basename, ifile, verbose=False)
                    except:
                        print(f'Downloading {basename} failed')
                        bad_files.append(basename)
                        # Remove from sub_files
                        sub_files.remove(ifile)
                        continue
                    
            if not local:
                print("All Done!")

            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                chunksize = len(sub_files) // n_cores if len(sub_files) // n_cores > 0 else 1
                answers = list(tqdm(executor.map(map_fn, basefiles,
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
                if f_prev is not None and kk == int(prev_meta['kk'].max())+1:
                    try:
                        f_h5.create_dataset('fields', data=f_prev['fields'][:], 
                                        compression="gzip", chunks=True,
                                        maxshape=(None, fields.shape[1], fields.shape[2]))
                    except:
                        embed(header='398 of v4')
                    f_h5.create_dataset('inpainted_masks', data=f_prev['inpainted_masks'][:],
                                        compression="gzip", chunks=True,
                                        maxshape=(None, inpainted_masks.shape[1], inpainted_masks.shape[2]))
                # Resize
                for key in ['fields', 'inpainted_masks']:
                    f_h5[key].resize((f_h5[key].shape[0] + fields.shape[0]), axis=0)
                # Fill
                f_h5['fields'][-fields.shape[0]:] = fields
                f_h5['inpainted_masks'][-fields.shape[0]:] = inpainted_masks
        
            # Remove em
            if not debug and not local:
                for ifile in sub_files:
                    basename = os.path.basename(ifile)
                    os.remove(basename)

            # Push to s3?
            if intermediate_s3:
                print("Pushing to s3")
                ulmo_io.upload_file_to_s3(save_path, s3_filename)

            # Metadata
            df = pandas.DataFrame(metadata)
            df['kk'] = kk
            df['nimgs'] = f_h5['fields'].shape[0]
            # Save where we are at
            csv_filename = 'curr_metadata_{}.csv'.format(year)
            df.to_csv(csv_filename, index=False)
            s3_csv_filename = 's3://modis-l2/TMP/curr_metadata_{}.csv'.format(year)
            ulmo_io.upload_file_to_s3(csv_filename, s3_csv_filename)


        # Metadata
        columns = ['filename', 'row', 'column', 'latitude', 'longitude', 
                'clear_fraction']
        dset = f_h5.create_dataset('metadata', data=metadata.astype('S'))
        dset.attrs['columns'] = columns
        # Close
        f_h5.close() 

        # Save me
        print("Pushing to s3")
        ulmo_io.upload_file_to_s3(save_path, s3_filename)

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
        modis_table['datetime'] = modis_utils.times_from_filenames(
            basefiles, ioff=10, toff=1)
        modis_table['ex_filename'] = s3_filename

        # Bad files
        bad_csv_file = f'bad_{year}.csv'
        df_bad = pandas.DataFrame()
        df_bad['filename'] = bad_files
        df_bad.to_csv(bad_csv_file, index=False)
        # s3
        s3_bad_file = 's3://modis-l2/TMP/'+bad_csv_file
        ulmo_io.upload_file_to_s3(bad_csv_file, s3_bad_file)

        # Vet
        assert cat_utils.vet_main_table(modis_table)

        # Save
        modis_tables.append(modis_table)

        # End loop on year

    # Concat
    modis_table = pandas.concat(modis_tables)

    # Vet
    assert cat_utils.vet_main_table(modis_table)

    # Final write
    ulmo_io.write_main_table(modis_table, tbl_file)
    
    #print("Run this:  s3 put {} s3://modis-l2/Extractions/{}".format(
    #    save_path, save_path))
    #process = subprocess.run(['s4cmd', '--force', '--endpoint-url',
    #    'https://s3.nautilus.optiputer.net', 'put', save_path, 
    #    s3_filename])

def modis_20s_preproc(debug=False, n_cores=20):
    """Pre-process the files

    Args:
        n_cores (int, optional): Number of cores to use
    """
    tbl_20s_file = 's3://modis-l2/Tables/MODIS_L2_20202021.parquet'
    modis_tbl = ulmo_io.load_main_table(tbl_20s_file)
    # Reset index
    modis_tbl.reset_index(drop=True, inplace=True)

    # Pre-process 
    modis_tbl = pp_utils.preproc_tbl(modis_tbl, 1., 
                                     's3://modis-l2',
                                     preproc_root='standard',
                                     inpainted_mask=True,
                                     use_mask=True,
                                     debug=debug,
                                     remove_local=False,
                                     nsub_fields=10000,
                                     n_cores=n_cores)
    # Vet
    assert cat_utils.vet_main_table(modis_tbl)

    # Final write
    if not debug:
        ulmo_io.write_main_table(modis_tbl, tbl_20s_file)

def slurp_tables(debug=False):
    tbl_20s_file = 's3://modis-l2/Tables/MODIS_L2_20202021.parquet'
    full_tbl_file = 's3://modis-l2/Tables/MODIS_SSL_96clear.parquet'

    # Load
    modis_20s_tbl = ulmo_io.load_main_table(tbl_20s_file)
    modis_full = ulmo_io.load_main_table(full_tbl_file)

    # Strip original if it is there..
    bad = modis_full.UID.values == modis_full.iloc[0].UID
    bad[0] = False
    modis_full = modis_full[~bad].copy()

    # Rename ulmo_pp_type
    modis_20s_tbl.rename(columns={'pp_type':'ulmo_pp_type'}, inplace=True)

    # Deal with filenames
    filenames = []
    for ifile in modis_20s_tbl.filename:
        filenames.append(os.path.basename(ifile))
    modis_20s_tbl['filename'] = filenames

    # Deal with pp_filenames
    pp_filenames = []
    for ifile in modis_20s_tbl.pp_file:
        pp_filenames.append(os.path.basename(ifile.replace('standard', 'std')))
    modis_20s_tbl['pp_file'] = pp_filenames

    # Fill up the new table with dummy values
    for key in modis_full.keys():
        if key not in modis_20s_tbl.keys():
            modis_20s_tbl[key] = modis_full[key].values[0]

    # Generate new UIDs
    modis_20s_tbl['UID'] = modis_utils.modis_uid(modis_20s_tbl)

    # Drop unwanted
    for key in modis_20s_tbl.keys():
        if key not in modis_full.keys():
            modis_20s_tbl.drop(key, axis=1, inplace=True)

    # Concat
    modis_full = pandas.concat([modis_full, modis_20s_tbl],
                               ignore_index=True)

    if debug:
        embed(header='597 of v4')

    # Vet
    assert cat_utils.vet_main_table(modis_full, cut_prefix='ulmo_')

    # Final write
    if not debug:
        ulmo_io.write_main_table(modis_full, full_tbl_file)



def cut_96(debug=False):
    full_tbl_file = 's3://modis-l2/Tables/MODIS_SSL_96clear.parquet'

    # Load
    modis_full = ulmo_io.load_main_table(full_tbl_file)

    # Cut
    cut = modis_full.clear_fraction < 0.04
    modis_full = modis_full[cut].copy()

    # Vet
    assert cat_utils.vet_main_table(modis_full, cut_prefix='ulmo_')

    # Final write
    if not debug:
        ulmo_io.write_main_table(modis_full, full_tbl_file)

def modis_ulmo_evaluate(debug=False):

    # Load
    tbl_20s_file = 's3://modis-l2/Tables/MODIS_L2_20202021.parquet'
    modis_tbl = ulmo_io.load_main_table(tbl_20s_file)

    # Grab 2020 and 2021
    if debug:
        embed(header='591 of v4')

    # Evaluate
    modis_tbl = ulmo_evaluate.eval_from_main(modis_tbl)

    # Write 
    assert cat_utils.vet_main_table(modis_tbl)

    if not debug:
        ulmo_io.write_main_table(modis_tbl, tbl_20s_file)


def calc_dt40(debug=False, local=False):

    # Table
    if local:
        tbl_file = os.path.join(os.getenv('SST_OOD'),
                                'MODIS_L2', 'Tables', 
                                'MODIS_SSL_96clear.parquet')
    else:
        tbl_file = 's3://modis-l2/Tables/MODIS_SSL_96clear.parquet'
    modis_tbl = ulmo_io.load_main_table(tbl_file)
    modis_tbl['DT40'] = 0.

    # Fix s3 in 2020
    new_pp_files = []
    for pp_file in modis_tbl.pp_file:
        if 's3' not in pp_file:
            new_pp_files.append('s3://modis-l2/PreProc/'+pp_file)
        else:
            new_pp_files.append(pp_file)
    modis_tbl['pp_file'] = new_pp_files
    
    # Grab the list
    preproc_files = np.unique(modis_tbl.pp_file.values)

    # Loop on files
    if debug:
        embed(header='659 of v4')
    for pfile in preproc_files:
        basename = os.path.basename(pfile)
        if local:
            basename = os.path.join(os.getenv('SST_OOD'),
                                'MODIS_L2', 'PreProc', basename) 
        else:
            # Download?
            if not os.path.isfile(basename):
                ulmo_io.download_file_from_s3(basename, pfile)

        # Open me
        print(f"Starting on: {basename}")
        f = h5py.File(basename, 'r')

        # Load it all
        DT40(f, modis_tbl, pfile, itype='valid')
        if 'train' in f.keys():
            DT40(f, modis_tbl, pfile, itype='train')

        # Remove 
        if not debug and not local:
            os.remove(basename)

        if debug:
            break

    # Vet
    assert cat_utils.vet_main_table(modis_tbl, cut_prefix='ulmo_')

    # Save
    if not debug:
        # To s3
        s3_file = 's3://modis-l2/Tables/MODIS_SSL_96clear.parquet'
        ulmo_io.write_main_table(modis_tbl, s3_file)

    print("All done")

def DT40(f:h5py.File, modis_tbl:pandas.DataFrame, 
         pfile:str, itype:str='train'):
    # Grab the data
    fields = f[itype][:]
    T_90 = np.percentile(fields[:, 0, 32-20:32+20, 32-20:32+20], 
        90., axis=(1,2))
    T_10 = np.percentile(fields[:, 0, 32-20:32+20, 32-20:32+20], 
        90., axis=(1,2))
    DT_40 = T_90 - T_10
    # Fill
    ppt = 0 if itype == 'valid' else 1
    idx = (modis_tbl.pp_file == pfile) & (modis_tbl.pp_type == ppt)
    pp_idx = modis_tbl[idx].pp_idx.values
    modis_tbl.loc[idx, 'DT40'] = DT_40[pp_idx]
    return 

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
    parser.add_argument('--local', default=False, action='store_true',
                        help='Local?')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Clobber existing files')
    parser.add_argument("--outfile", type=str, 
                        help="Path to output file")
    parser.add_argument("--umap_file", type=str, 
                        help="Path to UMAP pickle file for analysis")
    parser.add_argument("--table_file", type=str, 
                        help="Path to Table file")
    parser.add_argument("--ncpu", type=int, help="Number of CPUs")
    parser.add_argument("--years", type=str, help="Years to analyze")
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

    # python ssl_modis_v4.py --func_flag extract_new --ncpu 20 --local --years 2020 --debug
    if args.func_flag == 'extract_new':
        ncpu = args.ncpu if args.ncpu is not None else 10
        years = [int(item) for item in args.years.split(',')] if args.years is not None else [2020,2021]
        extract_modis(debug=args.debug, n_cores=ncpu, local=args.local, years=years)

    # python ssl_modis_v4.py --func_flag preproc --debug
    if args.func_flag == 'preproc':
        modis_20s_preproc(debug=args.debug)

    # python ssl_modis_v4.py --func_flag slurp_tables --debug
    if args.func_flag == 'slurp_tables':
        slurp_tables(debug=args.debug)

    # python ssl_modis_v4.py --func_flag cut_96 --debug
    if args.func_flag == 'cut_96':
        cut_96(debug=args.debug)

    # python ssl_modis_v4.py --func_flag ulmo_evaluate --debug
    if args.func_flag == 'ulmo_evaluate':
        modis_ulmo_evaluate(debug=args.debug)

    # python ssl_modis_v4.py --func_flag ssl_evaluate --debug
    if args.func_flag == 'ssl_evaluate':
        main_ssl_evaluate(args.opt_path, debug=args.debug)
        
    # python ssl_modis_v4.py --func_flag DT40 --debug --local
    if args.func_flag == 'DT40':
        calc_dt40(debug=args.debug, local=args.local)
