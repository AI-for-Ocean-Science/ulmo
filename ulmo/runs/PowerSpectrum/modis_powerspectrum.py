import numpy as np
import os

import argparse

import h5py

from ulmo import io as ulmo_io
from ulmo.analysis import fft
from ulmo.utils import catalog as cat_utils

from IPython import embed

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("argument for training.")
    parser.add_argument("--task", type=str, help="function to execute: 'slopes'")
    parser.add_argument("--options", type=str, help="Options for the task")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    args = parser.parse_args()
    
    return args

def measure_slopes(pargs):
    # Load the table
    tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Init
    if pargs.options is not None:
        options = pargs.options.split(',')
    else:
        options = []
    if 'init' in options:
        modis_tbl['zonal_slope'] = 0.
        modis_tbl['merid_slope'] = 0.
        modis_tbl['zonal_slope_err'] = 0.
        modis_tbl['merid_slope_err'] = 0.

    train = modis_tbl.pp_type == 1
    valid = modis_tbl.pp_type == 0

    # Unique PreProc files
    pp_files = np.unique(modis_tbl.pp_file)
    if pargs.debug:
        pp_files = pp_files[-1:]
        embed(header='47 debug')

    # Loop me
    for pp_file in pp_files:
        if ('2020s' in options and (
            ('2020' not in pp_file) or ('2021' not in pp_file))):
            continue
        # Open
        basefile = os.path.basename(pp_file)
        if not os.path.isfile(basefile):
            print(f"Downloading {pp_file} (this is *much* faster than s3 access)...")
            ulmo_io.download_file_from_s3(basefile, pp_file)
        pp_hf = h5py.File(basefile, 'r')

        if 'train' in pp_hf.keys():
            do_train = True
        else:
            do_train = False

        # Valid
        data1, data2, slopes, data4  = fft.process_preproc_file(
            pp_hf, key='valid') #, debug=pargs.debug

        # Save
        pidx = modis_tbl.pp_file == pp_file
        valid_idx = valid & pidx
        modis_tbl.loc[valid_idx, 'zonal_slope'] = slopes[:, 1]  # large
        modis_tbl.loc[valid_idx, 'zonal_slope_err'] = slopes[:, 2]  # large
        modis_tbl.loc[valid_idx, 'merid_slope'] = slopes[:, 4]  # large
        modis_tbl.loc[valid_idx, 'merid_slope_err'] = slopes[:, 5]  # large

        # Train
        if do_train:
            data1, data2, slopes, data4  = fft.process_preproc_file(
                pp_hf, key='train') #, debug=pargs.debug
            train_idx = train & pidx
            modis_tbl.loc[train_idx, 'zonal_slope'] = slopes[:, 1]  # large
            modis_tbl.loc[train_idx, 'zonal_slope_err'] = slopes[:, 2]  # large
            modis_tbl.loc[train_idx, 'merid_slope'] = slopes[:, 4]  # large
            modis_tbl.loc[train_idx, 'merid_slope_err'] = slopes[:, 5]  # large
        
        pp_hf.close()

        # Cleanup
        print(f"Done with {basefile}.  Cleaning up")
        if not pargs.debug:
            os.remove(basefile)

    # Vet
    assert cat_utils.vet_main_table(modis_tbl, cut_prefix='modis_')

    # Final write
    if not pargs.debug:
        ulmo_io.write_main_table(modis_tbl, tbl_file) 



if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    
    # run the 'main_train()' function.
    #
    # 2020s
    # python modis_powerspectrum.py --task slopes --options 2020s --debug 
    if args.task == 'slopes':
        print("Powerlaw measurements start.")
        measure_slopes(args)
        print("PowerLaw Ends.")
    