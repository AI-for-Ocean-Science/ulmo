""" module for vet of the SST L3S dataset
"""
import os
import numpy as np

import time
import h5py
import numpy as np
import argparse

import pandas
import datetime
#from datetime import datetime, timedelta

from matplotlib import pyplot as plt
import seaborn as sns

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils

from ulmo.sst_l3s import extract


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

    # Add L3S data
    l3s['row'] = ((90 - l3s['VIIRS_lat']) * (9000 / 180)).astype(int)
    l3s['col'] = ((l3s['VIIRS_lon'] + 180) * (18000 / 360)).astype(int)
    l3s['lat'] = l3s['VIIRS_lat']
    l3s['lon'] = l3s['VIIRS_lon']

    base_datetime = pandas.to_datetime(l3s['VIIRS_datetime']).dt.date.astype(str) + ' 01:30:00'
    base_datetime = pandas.to_datetime(base_datetime, format='%Y-%m-%d %H:%M:%S')
    l3s['datetime'] = (base_datetime + pandas.to_timedelta(l3s['VIIRS_lon'] * 4, unit='minutes')).dt.round('S')

    #embed(header='56 of sst_l3s_vet.py')
    l3s['ex_filename'] = (
        '/Volumes/Aqua-1/Hackathon/daily/l3s_fields/' +
        pandas.to_datetime(l3s['VIIRS_datetime']).dt.year.astype(str) +
        '/' +
        pandas.to_datetime(l3s['VIIRS_datetime']).dt.strftime('%j').astype(str) +
        '/' +
        pandas.to_datetime(l3s['VIIRS_datetime']).dt.year.astype(str) +
        pandas.to_datetime(l3s['VIIRS_datetime']).dt.strftime('%m').astype(str) +
        pandas.to_datetime(l3s['VIIRS_datetime']).dt.strftime('%d').astype(str) +
        '120000-STAR-L3S_GHRSST-SSTsubskin-LEO_Daily-ACSPO_V2.80-v02.0-fv01.0.nc'
    )

    # Check the table -- it should complain about missing required keys
    cat_utils.vet_main_table(l3s, cut_prefix='VIIRS_')

    # Write
    ulmo_io.write_main_table(l3s, l3s_viirs_tbl_file)

# EXTRACTION
def l3s_viirs_extract(tbl_file:str, 
                      root_file=None, 
                      preproc_root='l3s_viirs', 
                      debug=True):
    """ Perform the extraction for the L3S dataset

    Args:
        tbl_file (str): table file (s3)
        root_file (_type_, optional): 
            Output filename. Defaults to None.
        preproc_root (str, optional): 
            Defines the options for pre-processing. 
            Defaults to 'l3s_viirs'.
        debug (bool, optional): If True, perform
            a limited extraction as a test. Defaults to False.
    """

    # Giddy up (will take a bit of memory!)
    l3s_table = ulmo_io.load_main_table(tbl_file)

    if debug:
        # Cut down to the first month
        gd_date = l3s_table.datetime <= datetime.datetime(2012,2,2)
        l3s_table = l3s_table[gd_date]
        debug_local = True

    if debug:
        root_file = 'L3S_VIIRS144_test_preproc.h5'
    else:
        if root_file is None:
            root_file = 'L3S_VIIRS144_preproc.h5'

    #embed(header='105 sst_l3s_vet.py')

    # Setup
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://sst-l3s/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    print(f"Outputting to: {pp_s3_file}")

    # Run it
    if debug_local:
        pp_s3_file = 's3://sst-l3s/PreProc/tst.h5'
    # Check indices
    l3s_table.reset_index(drop=True, inplace=True)
    assert np.all(np.arange(len(l3s_table)) == l3s_table.index)
    # Do it
    #if debug:
    #    embed(header='210 of llc viirs')
    extract.preproc_for_analysis(l3s_table, 
                                 pp_local_file,
                                 preproc_root=preproc_root,
                                 s3_file=pp_s3_file,
                                 override_RAM=True)
    # Vet
    assert cat_utils.vet_main_table(l3s_table, cut_prefix=['VIIRS_'])

    # Final write
    if debug:
        ulmo_io.write_main_table(l3s_table, 'tmp.parquet', to_s3=False)
    else:
        ulmo_io.write_main_table(l3s_table, tbl_file)
    print("You should probably remove the PreProc/ folder")
    

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the VIIRS images
    if flg & (2**0):
        init_l3s_tbl()

    # Generate the VIIRS images
    if flg & (2**1):
        l3s_viirs_extract(l3s_viirs_tbl_file, debug=True)


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
