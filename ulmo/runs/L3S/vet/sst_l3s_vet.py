""" module for vet of the SST L3S dataset
"""
import os
import numpy as np

import time
import h5py
import numpy as np
import argparse

import pandas

from matplotlib import pyplot as plt
import seaborn as sns

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils
from ulmo.preproc import utils as pp_utils

from ulmo.preproc import io as pp_io 
from ulmo.modis import utils as modis_utils
from ulmo.modis import extract as modis_extract
from ulmo.analysis import evaluate as ulmo_evaluate 

from IPython import embed

l3s_viirs_tbl_file = 's3://sst-l3s/Tables/SST_L3S_VIIRS.parquet'
viirs_tbl_file = 's3://viirs/Tables/VIIRS_all_98clear_std.parquet'

def init_l3s_tbl():
    # Load VIIRS table
    viirs = ulmo_io.load_main_table(viirs_tbl_file)

    # Copy
    l3s = pandas.DataFrame()

    # Save VIIRS info
    viirs_keys = ['row', 'col', 'UID', 'LL', 'pp_file', 
                'pp_idx', 'T90', 'T10', 'DT', 'pp_type',
                'Tmin', 'Tmax',
                'clear_fraction', 'datetime', 'filename',
                'ex_filename', 'lat', 'lon']
    # Generate the L3S table
    for key in viirs_keys:
        l3s[f'VIIRS_{key}'] = viirs[key]

    # Check the table -- it should complain about missing required keys
    cat_utils.vet_main_table(l3s, cut_prefix='VIIRS_')

    # Write
    ulmo_io.write_main_table(l3s, l3s_viirs_tbl_file)

# EXTRACTION
def llc_viirs_extract(tbl_file:str, 
                      root_file=None, dlocal=True, 
                      preproc_root='llc_144', 
                      debug=False):

    # Giddy up (will take a bit of memory!)
    llc_table = ulmo_io.load_main_table(tbl_file)

    if debug:
        # Cut down to first 2 days
        uni_date = np.unique(llc_table.datetime)
        gd_date = llc_table.datetime <= uni_date[1]
        llc_table = llc_table[gd_date]
        debug_local = True

    if debug:
        root_file = 'LLC_VIIRS144_test_preproc.h5'
    else:
        if root_file is None:
            root_file = 'LLC_VIIRS144_preproc.h5'

    # Setup
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    print(f"Outputting to: {pp_s3_file}")

    # Run it
    #if debug_local:
    #    pp_s3_file = None  
    # Check indices
    llc_table.reset_index(drop=True, inplace=True)
    assert np.all(np.arange(len(llc_table)) == llc_table.index)
    # Do it
    if debug:
        embed(header='210 of llc viirs')
    extract.preproc_for_analysis(llc_table, 
                                 pp_local_file,
                                 fixed_km=144.,
                                 preproc_root=preproc_root,
                                 s3_file=pp_s3_file,
                                 #debug=debug,
                                 dlocal=dlocal,
                                 override_RAM=True)
    # Vet
    assert cat_utils.vet_main_table(llc_table, cut_prefix=['viirs_', 'MODIS_'])

    # Final write
    if not debug:
        ulmo_io.write_main_table(llc_table, tbl_file)
    print("You should probably remove the PreProc/ folder")
    

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the VIIRS images
    if flg & (2**0):
        init_l3s_tbl()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Init L3S table
    else:
        flg = sys.argv[1]

    main(flg)

# Generate the table
# python -u sst_l3s_vet.py 1
