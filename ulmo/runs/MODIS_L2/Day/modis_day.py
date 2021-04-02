""" Module for MODIS analysis on daytime SST"""
import os
import numpy as np

import pandas

from ulmo import io as ulmo_io
from ulmo.preproc import io as pp_io 
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.utils import catalog as cat_utils

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from IPython import embed

tbl_file = 's3://modis-l2/Tables/MODIS_L2_day_std.parquet'


def modis_day_extract():
    # Pre-processing (and extraction) settings
    pdict = pp_io('standard')

    # Setup for preproc
    map_fn = partial(extract_file,
                     load_path=load_path,
                     field_size=(pargs.field_size, pargs.field_size),
                     CC_max=1.-pargs.clear_threshold / 100.,
                     qual_thresh=pargs.quality_threshold,
                     nadir_offset=pargs.nadir_offset,
                     temp_bounds=(pargs.temp_lower_bound, pargs.temp_upper_bound),
                     nrepeat=pargs.nrepeat,
                     inpaint=not pargs.no_inpaint,
                     debug=pargs.debug)


    llc_table = extract.preproc_for_analysis(llc_table, 
                                 pp_local_file,
                                 preproc_root=preproc_root,
                                 s3_file=pp_s3_file,
                                 dlocal=False,
                                 debug=debug)
    # Vet
    assert cat_utils.vet_main_table(llc_table, cut_prefix='modis_')

    # Final write
    if not debug:
        ulmo_io.write_main_table(llc_table, tbl_file)
    

def modis_evaluate(test=True, noise=False):

    if test:
        tbl_file = tbl_test_noise_file if noise else tbl_test_file
    else:
        raise IOError("Not ready for anything but testing..")
    
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Rename
    if 'LL' in llc_table.keys() and 'modis_LL' not in llc_table.keys():
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

    if flg & (2**3):
        modis_init_test(show=False, noise=True, localCC=False)
        #modis_init_test(show=True, noise=True, localCC=True)#, localM=False)

    if flg & (2**4):
        modis_extract(noise=True, debug=False)

    if flg & (2**5):
        modis_evaluate(noise=True)

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
    else:
        flg = sys.argv[1]

    main(flg)