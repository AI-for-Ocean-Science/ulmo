""" Module for Ulmo analysis on VIIRS 2013"""
import os
import numpy as np
import glob
import pandas
import h5py
from pkg_resources import resource_filename

import argparse

from sklearn.utils import shuffle

from ulmo import io as ulmo_io
from ulmo.preproc import io as pp_io 
from ulmo.preproc import utils as pp_utils
from ulmo.ssh import extract as ssh_extract
from ulmo.ssh import utils as ssh_utils
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.utils import catalog as cat_utils
from ulmo.ood import ood

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from IPython import embed

std_tbl_file = 's3://ssh/Tables/SSH_std.parquet'
s3_bucket = 's3://ssh'

def ssh_extraction(pargs, n_cores=15, 
                       nsub_files=300,
                       ndebug_files=10):
    """Extract *all* of the SSH data

    Args:
        debug (bool, optional): [description]. Defaults to False.
        n_cores (int, optional): Number of cores to use. Defaults to 20.
        nsub_files (int, optional): Number of sub files to process at a time. Defaults to 5000.
        ndebug_files (int, optional): [description]. Defaults to 0.
    """
    # 10 cores took 6hrs
    # 20 cores took 3hrs

    if pargs.debug:
        tbl_file = 's3://ssh/Tables/SSH_tst.parquet'
    else:
        tbl_file = std_tbl_file

    # TODO -- BP to figure out what goes on here
    #  and modify the JSON file.  Look down below at the
    #  call to extract_file()
    # Pre-processing (and extraction) settings
    pdict = pp_io.load_options('ssh_std')
    #embed(header='51 of ssh_run')
    
    # 2013 
    print("Grabbing the file list")
    path_files = os.path.join(os.getenv('SSH_DATA'), 'ssh*.nc')
    files = glob.glob(path_files)

    #ulmo_io.list_of_bucket_files('ssh')
    #all_ssh_files = glob.glob("/home/jovyan/sshdata_full/ssh*.nc") #ulmo_io.list_of_bucket_files('ssh')
    #for ifile in all_ssh_files:
    #    if 'SSH_Data_Files' in ifile:
    #        files.append(s3_bucket+'/'+ifile)

    # Output
    if pargs.debug:
        save_path = ('SSH'
                 '_{}clear_{}x{}_tst.h5'.format(
                     pdict['clear_threshold'], 
                     pdict['field_size'], 
                     pdict['field_size']))
    else:                                                
        save_path = ('SSH'
                 '_{}clear_{}x{}.h5'.format(
                     pdict['clear_threshold'], 
                     pdict['field_size'], 
                     pdict['field_size']))
    s3_filename = 's3://ssh/Extractions/{}'.format(save_path)

    if pargs.debug:
        # Grab 100 random
        files = shuffle(files, random_state=1234)
        files = files[:ndebug_files]  # 10%
        #files = files[:100]
        
    # Setup for preproc
    map_fn = partial(ssh_extract.extract_file,
                     field_size=(pdict['field_size'], 
                                 pdict['field_size']),
                     CC_max=1.-pdict['clear_threshold'] / 100.,
                     nadir_offset=pdict['nadir_offset'],
                     temp_bounds=tuple(pdict['temp_bounds']),
                     nrepeat=pdict['nrepeat'],
                     sub_grid_step=pdict['sub_grid_step'],
                     inpaint=False)

    # Local file for writing
    f_h5 = h5py.File(save_path, 'w')
    print("Opened local file: {}".format(save_path))
    
    nloop = len(files) // nsub_files + ((len(files) % nsub_files) > 0)
    metadata = None
    for kk in range(nloop):
        # Zero out
        fields, inpainted_masks = None, None
        #
        i0 = kk*nsub_files
        i1 = min((kk+1)*nsub_files, len(files))
        print('Files: {}:{} of {}'.format(i0, i1, len(files)))
        sub_files = files[i0:i1]
        
        #embed(header='108 of ssh_run')
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(sub_files) // n_cores if len(sub_files) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, sub_files,
                                             chunksize=chunksize), total=len(sub_files)))
        #embed(header='114 of ssh_run')
        # Trim None's
        answers = [f for f in answers if f is not None]
        fields = np.concatenate([item[0] for item in answers])
        inpainted_masks = np.concatenate([item[1] for item in answers])
        if metadata is None:
            metadata = np.concatenate([item[2] for item in answers])
        else:
            metadata = np.concatenate([metadata]+[item[2] for item in answers], axis=0)
        del answers
        #embed(header='124 of ssh_run')
        # Write
        if kk == 0:
            f_h5.create_dataset('fields', data=fields, 
                                compression="gzip", chunks=True,
                                maxshape=(None, fields.shape[1], fields.shape[2]))
            #f_h5.create_dataset('inpainted_masks', data=inpainted_masks,
            #                    compression="gzip", chunks=True,
            #                    maxshape=(None, inpainted_masks.shape[0], inpainted_masks.shape[0]))
        else:
            # Resize
            for key in ['fields']:#, 'inpainted_masks']:
                f_h5[key].resize((f_h5[key].shape[0] + fields.shape[0]), axis=0)
            # Fill
            f_h5['fields'][-fields.shape[0]:] = fields
            #f_h5['inpainted_masks'][-fields.shape[0]:] = inpainted_masks
    
    #embed(header='141 of ssh_run')
    # Metadata
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 
               'clear_fraction']
    dset = f_h5.create_dataset('metadata', data=metadata.astype('S'))
    dset.attrs['columns'] = columns
    # Close
    f_h5.close() 
    #embed(header='149 of ssh_run')
    # Table time
    ssh_table = pandas.DataFrame()
    ssh_table['filename'] = [item[0] for item in metadata]
    ssh_table['row'] = [int(item[1]) for item in metadata]
    ssh_table['col'] = [int(item[2]) for item in metadata]
    ssh_table['lat'] = [float(item[3]) for item in metadata]
    ssh_table['lon'] = [float(item[4]) for item in metadata]
    ssh_table['clear_fraction'] = [float(item[5]) for item in metadata]
    ssh_table['field_size'] = pdict['field_size']
    basefiles = [os.path.basename(ifile) for ifile in ssh_table.filename.values]
    ssh_table['datetime'] = ssh_utils.times_from_filenames(basefiles)#, ioff=-1, toff=0)
    ssh_table['ex_filename'] = s3_filename

    # Vet
    assert cat_utils.vet_main_table(ssh_table)

    # Final write
    ulmo_io.write_main_table(ssh_table, tbl_file)
    
    # Push to s3
    print("Pushing to s3")
    ulmo_io.upload_file_to_s3(save_path, s3_filename)
    #print("Run this:  s3 put {} s3://modis-l2/Extractions/{}".format(
    #    save_path, save_path))
    #process = subprocess.run(['s4cmd', '--force', '--endpoint-url',
    #    'https://s3.nautilus.optiputer.net', 'put', save_path, 
    #    s3_filename])


def ssh_preproc(pargs, n_cores=20, valid_fraction=0.95):
    """Pre-process the files

    Args:
        n_cores (int, optional): Number of cores to use
        valid_fraction (float, optional): 
            1-valid_fraction is the % of the (random) images used for training
    """
    if pargs.debug:
        tbl_file = 's3://ssh/Tables/SSH_tst.parquet'
        valid_fraction = 0.5
    else:
        tbl_file = std_tbl_file

    # Load Table
    ssh_tbl = ulmo_io.load_main_table(tbl_file)

    # Do it!
    ssh_tbl = pp_utils.preproc_tbl(ssh_tbl, 
                                   valid_fraction,
                                     's3://ssh',
                                     preproc_root='ssh_std',
                                     inpainted_mask=False,
                                     use_mask=False,
                                     nsub_fields=10000,
                                     n_cores=n_cores)
    # Vet
    assert cat_utils.vet_main_table(ssh_tbl)

    # Final write
    ulmo_io.write_main_table(ssh_tbl, tbl_file)

def ssh_cut_train(pargs, nvalid_train=600000, 
                  train_file='PreProc/SSH_100clear_32x32_train.h5',
                  preproc_file='PreProc/SSH_100clear_32x32.h5'):
    # Open
    f = h5py.File(preproc_file, 'r')

    # Grab the train images
    print("Loading training data")
    train = f['train'][...]
    train_meta = f['train_metadata'][...]
    clms = f['train_metadata'].attrs['columns']

    # Grab a random subset of valid
    print("Loading valid data")
    nvalid = f['valid'].shape[0]
    valid = f['valid'][...]
    # Cut
    idx = sorted(np.random.choice(nvalid, replace=False, size=nvalid_train))
    print("Valid train")
    valid_train = valid[idx]
    print("Starting Valid train meta")
    valid_meta = f['valid_metadata'][...]
    valid_train_meta = valid_meta[idx]

    # Write
    print(f"Writing: {train_file}...")
    with h5py.File(train_file, 'w') as f:
        # Validation
        f.create_dataset('valid', data=valid_train.astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', data=valid_train_meta)
        dset.attrs['columns'] = clms
        # Train
        f.create_dataset('train', data=train.astype(np.float32))
        dset = f.create_dataset('train_metadata', data=train_meta)
        dset.attrs['columns'] = clms
    print(f"Wrote: {train_file}...")

def ssh_train_flow(pargs, model='ssh-std'):
    """To be run on Nautilus Jupyter-Hub

    Args:
        pargs (_type_): _description_
        model (str, optional): _description_. Defaults to 'ssh-std'.
    """
    dpath = '/home/jovyan/Oceanography/SSH/Training/'
    datadir= os.path.join(dpath, 'SSH_std')
    model_file = os.path.join(resource_filename('ulmo', 'ssh'), 
                              'ssh_pae_model_std.json')
    preproc_file = os.path.join(dpath, 'PreProc', 
                                'SSH_100clear_32x32_train.h5')
    # Instantiate
    pae = ood.ProbabilisticAutoencoder.from_json(model_file, 
                                                filepath=preproc_file,
                                                datadir=datadir, logdir=datadir)
    # Load                                    
    pae.load_autoencoder()
    # Train
    pae.train_flow(n_epochs=10, batch_size=64, lr=2.5e-4, 
                   summary_interval=50, 
                   eval_interval=2500)  # 2000 may be better

    # Set to local stuff..
    pae.filepath['latents'] = 'SSH_std/SSH_100clear_32x32_train_latents.h5'
    pae.filepath['log_probs'] = 'SSH_std/SSH_100clear_32x32_train_log_probs.h5'

    # Plot
    pae.plot_log_probs(save_figure=True, logdir='SSH_std')

def ssh_evaluate(pargs, model='ssh-std'):
    """Evaluate the ssh data using Ulmo

    Args:
        debug (bool, optional): [description]. Defaults to False.
        model (str, optional): [description]. Defaults to 'modis-l2-std'.
    """

    if pargs.debug:
        tbl_file = 's3://ssh/Tables/SSH_tst.parquet'
    else:
        tbl_file = std_tbl_file

    # Load Ulmo
    ssh_tbl = ulmo_io.load_main_table(tbl_file)

    # Evaluate
    print("Starting evaluating..")
    ssh_tbl = ulmo_evaluate.eval_from_main(ssh_tbl, model=model)

    # Write 
    assert cat_utils.vet_main_table(ssh_tbl)
    ulmo_io.write_main_table(ssh_tbl, tbl_file)
    print("Done evaluating..")


def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("Running SSH")
    parser.add_argument("step", type=str, help="Step of the run")
    #parser.add_argument("--func_flag", type=str, help="flag of the function to be execute: 'train' or 'evaluate' or 'umap'.")
        # JFH Should the default now be true with the new definition.
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    # get the arguments
    pargs = parse_option()

    if pargs.step == 'extract':
        ssh_extraction(pargs)

    if pargs.step == 'preproc':
        ssh_preproc(pargs)

    if pargs.step == 'cut_for_training':
        ssh_cut_train(pargs)

    if pargs.step == 'train_flow':
        ssh_train_flow(pargs)

    if pargs.step == 'evaluate':
        ssh_evaluate(pargs)

# Extract
# python ssh_run.py extract --debug
# python ssh_run.py extract 

# Pre-process
# python ssh_run.py preproc --debug

# Cut for training
# python ssh_run.py cut_for_training

# Train flow
# python ssh_run.py train_flow

# Evaluate
# python ssh_run.py evaluate --debug