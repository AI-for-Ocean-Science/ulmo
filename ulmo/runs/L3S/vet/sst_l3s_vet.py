""" module for vet of the SST L3S dataset
"""
from email import header
from operator import mod
import os
from typing import IO
import numpy as np
import glob

import time
import h5py
import numpy as np
from tqdm.auto import trange
import argparse

import pandas
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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

    # Save VIIRS keys
    for key in ['row', 'col', 'UID', 'LL', 'pp_file', 'pp_idx', 'T90', 'T10', 'DT']:

    # Generate the L3S table
    embed(header='43 of sst_l3s_vet.py')

    # Check the table

    # Write
    ulmo_io.write_main_table(l3s, l3s_viirs_tbl_file

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
