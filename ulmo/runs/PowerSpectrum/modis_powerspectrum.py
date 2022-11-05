import numpy as np
import os
import time

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
    parser.add_argument("--tbl_file", type=str, help="Table to work on")
    parser.add_argument("--options", type=str, help="Options for the task")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    args = parser.parse_args()
    
    return args

def measure_slopes(pargs):
    # Load the table
    tbl_file = pargs.tbl_file if pargs.tbl_file is not None else 's3://modis-l2/Tables/MODIS_L2_std.parquet'
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
    
    # Reset pp_type
    if '2020s' in options:
        for year in ['2020', '2021']:
            in_year = modis_tbl.pp_file == f's3://modis-l2/PreProc/MODIS_R2019_{year}_95clear_128x128_preproc_std.h5'
            if 'ulmo_pp_type' not in modis_tbl.keys():
                raise ValueError("Need to adjust this for pp_type")
            modis_tbl.loc[in_year, 'ulmo_pp_type'] = 0
        train = modis_tbl.ulmo_pp_type == 1
        valid = modis_tbl.ulmo_pp_type == 0
    else: # Original
        train = modis_tbl.pp_type == 1
        valid = modis_tbl.pp_type == 0

    # Unique PreProc files
    pp_files = np.unique(modis_tbl.pp_file)
    if pargs.debug:
        pp_files = pp_files[-1:]

    # Loop me
    for pp_file in pp_files:
        tstart = time.time()
        if '2020s' in options:
            ok = ('2020' in pp_file) or ('2021' in pp_file)
            if not ok:
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
            pp_hf, key='valid', debug=pargs.debug)

        # Save
        pidx = modis_tbl.pp_file == pp_file
        valid_idx = valid & pidx

        if pargs.debug:
            tmp = np.zeros((np.sum(valid_idx), 6))
            tmp[0:100,:] = slopes
            slopes = tmp

        # Dang pandas loc
        modis_tbl.zonal_slope.values[valid_idx] = slopes[:, 1]  # large
        modis_tbl.zonal_slope_err.values[valid_idx] = slopes[:, 2]  # large
        modis_tbl.merid_slope.values[valid_idx] = slopes[:, 4]  # large
        modis_tbl.merid_slope_err.values[valid_idx] = slopes[:, 5]  # large

        # Train
        if do_train:
            data1, data2, slopes, data4  = fft.process_preproc_file(
                pp_hf, key='train', debug=pargs.debug)
            train_idx = train & pidx
            modis_tbl.zonal_slope.values[train_idx] = slopes[:, 1]  # large
            modis_tbl.zonal_slope_err.values[train_idx] = slopes[:, 2]  # large
            modis_tbl.merid_slope.values[train_idx] = slopes[:, 4]  # large
            modis_tbl.merid_slope_err.values[train_idx] = slopes[:, 5]  # large
            #modis_tbl.loc[train_idx, 'zonal_slope'] = slopes[:, 1]  # large
            #modis_tbl.loc[train_idx, 'zonal_slope_err'] = slopes[:, 2]  # large
            #modis_tbl.loc[train_idx, 'merid_slope'] = slopes[:, 4]  # large
            #modis_tbl.loc[train_idx, 'merid_slope_err'] = slopes[:, 5]  # large
        
        pp_hf.close()

        # Cleanup
        print(f"Done with {basefile}.  Cleaning up")
        print(get_time_string(time.time()-tstart))
        if pargs.debug:
            print('It will take:')
            print(get_time_string(slopes.shape[0]/100*(time.time()-tstart)))
        if not pargs.debug:
            os.remove(basefile)

    # Vet
    assert cat_utils.vet_main_table(modis_tbl, cut_prefix=['modis_', 'ulmo_'])

    # Final write
    if not pargs.debug:
        ulmo_io.write_main_table(modis_tbl, tbl_file) 

def get_time_string(codetime):
    """
    Utility function that takes the codetime and
    converts this to a human readable String.

    Args:
        codetime (`float`):
            Code execution time in seconds (usually the difference of two time.time() calls)

    Returns:
        `str`: A string indicating the total execution time
    """
    if codetime < 60.0:
        retstr = 'Execution time: {0:.2f}s'.format(codetime)
    elif codetime / 60.0 < 60.0:
        mns = int(codetime / 60.0)
        scs = codetime - 60.0 * mns
        retstr = 'Execution time: {0:d}m {1:.2f}s'.format(mns, scs)
    else:
        hrs = int(codetime / 3600.0)
        mns = int(60.0 * (codetime / 3600.0 - hrs))
        scs = codetime - 60.0 * mns - 3600.0 * hrs
        retstr = 'Execution time: {0:d}h {1:d}m {2:.2f}s'.format(hrs, mns, scs)
    return retstr

if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    
    # run the 'main_train()' function.
    #
    # 2020s
    # python modis_powerspectrum.py --task slopes --options 2020s --tbl_file s3://modis-l2/Tables/MODIS_SSL_v4.parquet --debug 
    # python modis_powerspectrum.py --task slopes --options 2020s --tbl_file /home/xavier/Projects/Oceanography/SST/MODIS_L2/Tables/MODIS_SSL_v4.parquet --debug 
    if args.task == 'slopes':
        print("Powerlaw measurements start.")
        measure_slopes(args)
        #
        print("PowerLaw Ends.")
        
    