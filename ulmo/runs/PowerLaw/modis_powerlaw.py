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
    parser.add_argument("--task", type=str, help="function to execute: 'powerlaw'")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    args = parser.parse_args()
    
    return args

def measure_powerlaw(pargs):
    # Load the table
    tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Init
    modis_tbl['zonal_slope'] = 0.
    modis_tbl['merid_slope'] = 0.

    train = modis_tbl.pp_type == 1
    valid = modis_tbl.pp_type == 0

    # Unique PreProc files
    pp_files = np.unique(modis_tbl.pp_file)
    if pargs.debug:
        pp_files = pp_files[0:1]

    # Loop me
    for pp_file in pp_files:
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
        data1, data2, data3, data4  = fft.process_preproc_file(
            pp_hf, key='valid') #, debug=pargs.debug

        # Save
        pidx = modis_tbl.pp_file == pp_file
        valid_idx = valid & pidx
        modis_tbl.loc[valid_idx, 'zonal_slope'] = data3[:, 1]  # large
        modis_tbl.loc[valid_idx, 'merid_slope'] = data3[:, 3]  # large

        # Train
        if do_train:
            data1, data2, data3, data4  = fft.process_preproc_file(
                pp_hf, key='train') #, debug=pargs.debug
            train_idx = train & pidx
            modis_tbl.loc[train_idx, 'zonal_slope'] = data3[:, 1]  # large
            modis_tbl.loc[train_idx, 'merid_slope'] = data3[:, 3]  # large
        
        pp_hf.close()

        # Cleanup
        print(f"Done with {basefile}.  Cleaning up")
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
    if args.task == 'powerlaw':
        print("Powerlaw measurements start.")
        measure_powerlaw(args)
        print("PowerLaw Ends.")
    