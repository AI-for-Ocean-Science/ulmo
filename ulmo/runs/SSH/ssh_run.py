""" Module for Ulmo analysis on VIIRS 2013"""
import os
import numpy as np

import pandas
import h5netcdf.legacyapi as h5py

import argparse

from sklearn.utils import shuffle

from ulmo import io as ulmo_io
from ulmo.preproc import io as pp_io 
from ulmo.preproc import utils as pp_utils
from ulmo.ssh import extract as ssh_extract
from ulmo.modis import utils as modis_utils
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.utils import catalog as cat_utils

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from IPython import embed

tbl_file = 's3://ssh/Tables/SSH_std.parquet'
s3_bucket = 's3://ssh'

def ssh_extraction(pargs, n_cores=20, 
                       nsub_files=5000,
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

    # TODO -- BP to figure out what goes on here
    #  and modify the JSON file
    # Pre-processing (and extraction) settings
    pdict = pp_io.load_options('ssh_std')
    #embed(header='51 of ssh_run')
    
    # 2013 
    print("Grabbing the file list")
    all_ssh_files = os.listdir("/home/jovyan/sshdata_mini") #ulmo_io.list_of_bucket_files('ssh')
    files = []
    for ifile in all_ssh_files:
        if 'SSH_Data_Files' in ifile:
            files.append(s3_bucket+'/'+ifile)

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
        
    embed(header='81 of ssh_run')
    
    # Setup for preproc
    map_fn = partial(ssh_extract.extract_file,
                     field_size=(pdict['field_size'], pdict['field_size']),
                     CC_max=1.-pdict['clear_threshold'] / 100.,
                     nadir_offset=pdict['nadir_offset'],
                     temp_bounds=tuple(pdict['temp_bounds']),
                     nrepeat=pdict['nrepeat'],
                     sub_grid_step=pdict['sub_grid_step'],
                     inpaint=True)

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

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(sub_files) // n_cores if len(sub_files) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, sub_files,
                                             chunksize=chunksize), total=len(sub_files)))

        # Trim None's
        answers = [f for f in answers if f is not None]
        fields = np.concatenate([item[0] for item in answers])
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
    

    # Metadata
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 
               'clear_fraction']
    dset = f_h5.create_dataset('metadata', data=metadata.astype('S'))
    dset.attrs['columns'] = columns
    # Close
    f_h5.close() 

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
    ssh_table['datetime'] = modis_utils.times_from_filenames(basefiles, ioff=-1, toff=0)
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


def viirs_2013_preproc(debug=False, n_cores=20):
    """Pre-process the files

    Args:
        n_cores (int, optional): Number of cores to use
    """
    if debug:
        tbl_file = 's3://viirs/Tables/VIIRS_2013_tst.parquet'
    else:
        tbl_file = tbl_file_2013
    viirs_tbl = ulmo_io.load_main_table(tbl_file)
    viirs_tbl = pp_utils.preproc_tbl(viirs_tbl, 1., 
                                     's3://viirs',
                                     preproc_root='viirs_std',
                                     inpainted_mask=True,
                                     use_mask=True,
                                     nsub_fields=10000,
                                     n_cores=n_cores)
    # Vet
    assert cat_utils.vet_main_table(viirs_tbl)

    # Final write
    ulmo_io.write_main_table(viirs_tbl, tbl_file)

def viirs_2013_evaluate(debug=False, model='modis-l2-std'):
    """Evaluate the VIIRS 2013 data using Ulmo

    Args:
        debug (bool, optional): [description]. Defaults to False.
        model (str, optional): [description]. Defaults to 'modis-l2-std'.
    """

    if debug:
        tbl_file = 's3://viirs/Tables/VIIRS_2013_tst.parquet'
    else:
        tbl_file = tbl_file_2013

    # Load Ulmo
    viirs_tbl = ulmo_io.load_main_table(tbl_file)

    # Evaluate
    print("Starting evaluating..")
    viirs_tbl = ulmo_evaluate.eval_from_main(viirs_tbl, model=model)

    # Write 
    assert cat_utils.vet_main_table(viirs_tbl)
    ulmo_io.write_main_table(viirs_tbl, tbl_file)
    print("Done evaluating..")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # VIIRS download
    if flg & (2**0):
        viirs_get_data_into_s3(debug=False)

    # VIIRS extract test
    if flg & (2**1):
        viirs_extract_2013(debug=True, n_cores=10, nsub_files=2000, ndebug_files=5000)

    # VIIRS extract
    if flg & (2**2):
        viirs_extract_2013(n_cores=20, nsub_files=5000)

    # VIIRS preproc test
    if flg & (2**3):
        viirs_2013_preproc(debug=True, n_cores=10)

    # VIIRS preproc for reals
    if flg & (2**4):
        viirs_2013_preproc(n_cores=20)

    # VIIRS eval test
    if flg & (2**5): # 32
        viirs_2013_evaluate(debug=True)

    # VIIRS eval 
    if flg & (2**6): # 64
        viirs_2013_evaluate()


    # MODIS pre-proc
    #if flg & (2**2):
    #    modis_day_evaluate()


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

# Run it

# Extract
# python ssh_run.py extract --debug
