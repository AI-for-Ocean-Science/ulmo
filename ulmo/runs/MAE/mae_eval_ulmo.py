""" Module to run all analysis related to fixed 144km Uniform sampling of LLC 
 144km is equivalent to 64 pixels at VIIRS sampling binned by 3
"""
import os
import numpy as np

import h5py

from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate
from ulmo.utils import catalog as cat_utils
from ulmo.mae import mae_utils

from IPython import embed

llc_tst_file = 's3://llc/Tables/test_uniform144_r0.5_test.parquet'
llc_full_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
llc_nonoise_file = 's3://llc/Tables/LLC_uniform144_r0.5_nonoise.parquet'

# MAE
mae_tst_nonoise_file = 's3://llc/mae/Tables/MAE_uniform144_test.parquet'
mae_nonoise_file = 's3://llc/mae/Tables/MAE_uniform144_nonoise.parquet'
mae_img_path = 's3://llc/mae/PreProc'

def gen_mae_tbl(tbl_file:str, outfile:str):
    """ Generate an MAE table

    Args:
        tbl_file (str): _description_
        outfile (str): _description_
    """
    # Load
    orig_table = ulmo_io.load_main_table(tbl_file)

    # New Table
    mae_tbl = orig_table.copy()

    # Save LL
    mae_tbl['LL_orig'] = mae_tbl.LL

    # Vet
    chk, disallowed_keys = cat_utils.vet_main_table(
        mae_tbl, return_disallowed=True)
    assert disallowed_keys[0] == 'LL_orig'

    # Write
    ulmo_io.write_main_table(mae_tbl, outfile)


def mae_ulmo_evaluate(tbl_file:str, img_files:list,
                  clobber=False, debug=False, 
                  model='viirs-98'):
    """ Run Ulmo on MAE cutouts with the given model
    """
    
    # Load
    mae_table = ulmo_io.load_main_table(tbl_file)
    print("Loaded MAE table")

    # Debug?
    if debug:
        f = h5py.File(os.path.basename(img_files[0]), 'r')
        nimg = f['valid'].shape[0]
        # Chop down table
        gd_tbl = (mae_table.pp_idx >= 0) & (mae_table.pp_idx < nimg)
        mae_table = mae_table[gd_tbl].copy()

    # Loop on eval_files
    for img_file in img_files:
        # Fuss with LL
        t_per, p_per = mae_utils.parse_mae_img_file(img_file)
        LL_metric = f'LL_t{t_per}_p{p_per}'

        # Already exist?
        if LL_metric in mae_table.keys() and not clobber:
            print(f"LL metric = {LL_metric} already evaluated.  Skipping..")
            continue

        # Reset pp_file for the evaluation that follows
        mae_table.pp_file = img_file

        # Evaluate
        embed(header='81 of mae_eval_ulmo')
        mae_table = ulmo_evaluate.eval_from_main(mae_table,
                                 model=model)

        # Save LL
        mae_table[f'LL_t{t_per}_p{p_per}'] = mae_table.LL

        # Could save after each eval..

    # Vet
    chk, disallowed_keys = cat_utils.vet_main_table(
        mae_table, return_disallowed=True)
    for key in disallowed_keys:
        assert key[0:2] == 'LL'

    # Write 
    if debug:
        tbl_file = tbl_file.replace('.parquet', '_small.parquet')
        embed(header='99 of mae_eval_ulmo')
    ulmo_io.write_main_table(mae_table, tbl_file)


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the MAE Table
    if flg & (2**0):
        #gen_mae_tbl(llc_nonoise_file, mae_nonoise_file)
        # Debug table
        gen_mae_tbl(llc_tst_file, mae_tst_nonoise_file)

    # Evaluate LL with ulmo
    if flg & (2**1):

        # Ulmo model
        model='viirs-98'
        debug = True

        # Image parameters -- (train_percenntage, patch_percentage)
        img_pers = [(10, 10)]  

        # Generate the file names
        img_files = []
        for img_per in img_pers:
            img_file = mae_utils.img_filename(img_per[0], img_per[1])
            if debug:
                img_file = img_file.replace('.h5', '_small.h5')
            img_files.append(img_file)
        
        # Run
        mae_ulmo_evaluate(mae_nonoise_file, img_files,
                          model=model, clobber=False, debug=debug)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup Table
        #flg += 2 ** 1  # 2 -- Evaluate 
    else:
        flg = sys.argv[1]

    main(flg)

# Generate the table(s)
# python -u mae_eval_ulmo.py 1

# Evaluate
# python -u mae_eval_ulmo.py 2
