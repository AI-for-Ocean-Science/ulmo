""" Module for LLC Ulmo analyses"""
import os
import numpy as np

import pandas

from ulmo.llc import extract 
from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.utils import catalog as cat_utils


from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky

from IPython import embed

def ulmo_evaluate(noise=False, tbl_file=None, rename=True):

    if tbl_file is None:
        if test:
            tbl_file = tbl_test_noise_file if noise else tbl_test_file
        else:
            raise IOError("Not ready for anything but testing..")
    
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Rename
    if rename and 'LL' in llc_table.keys() and 'modis_LL' not in llc_table.keys():
        llc_table = llc_table.rename(
            columns=dict(LL='modis_LL'))

    # Evaluate
    llc_table = ulmo_evaluate.eval_from_main(llc_table)

    # Write 
    assert cat_utils.vet_main_table(llc_table, cut_prefix='modis_')
    ulmo_io.write_main_table(llc_table, tbl_file)

def u_add_velocities():
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)
    
    # Velocities
    extract.velocity_stats(llc_table)

    # Write 
    ulmo_io.write_main_table(llc_table, tbl_file)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

        # MMT/MMIRS
    if flg & (2**0):
        modis_init_test(show=True)

    if flg & (2**1):
        modis_extract()

    if flg & (2**2):
        modis_evaluate()

    # 2012 + noise
    if flg & (2**3):
        modis_init_test(show=False, noise=True, localCC=False)
        #modis_init_test(show=True, noise=True, localCC=True)#, localM=False)

    if flg & (2**4):
        modis_extract(noise=True, debug=False)

    if flg & (2**5):
        modis_evaluate(noise=True)

    if flg & (2**6):  # Debuggin
        modis_evaluate(tbl_file='s3://llc/Tables/test2_modis2012.parquet')

    if flg & (2**7):  
        modis_evaluate(tbl_file='s3://llc/Tables/LLC_modis_noise_track.parquet')


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup coords
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate
        #flg += 2 ** 3  # 8 -- Init test + noise
        #flg += 2 ** 4  # 16 -- Extract + noise
        #flg += 2 ** 5  # 32 -- Evaluate + noise
        #flg += 2 ** 6  # 64 -- Evaluate debug run
        flg += 2 ** 7  # 128 -- Katharina's first noise try
    else:
        flg = sys.argv[1]

    main(flg)
