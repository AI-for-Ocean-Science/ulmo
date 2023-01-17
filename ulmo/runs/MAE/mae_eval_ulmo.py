""" Module to run all analysis related to fixed 144km Uniform sampling of LLC 
 144km is equivalent to 64 pixels at VIIRS sampling binned by 3
"""
import os
import numpy as np

import pandas

from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate
from ulmo.utils import catalog as cat_utils
from ulmo.mae import mae_utils

from IPython import embed

llc_tst_file = 's3://llc/Tables/test_uniform144_r0.5_test.parquet'
llc_full_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
llc_nonoise_file = 's3://llc/Tables/LLC_uniform144_r0.5_nonoise.parquet'

# MAE
mae_nonoise_file = 's3://llc/MAE/Tables/MAE_uniform144_nonoise.parquet'
mae_img_path = 's3://llc/MAE/PreProc'

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
    chk, disallowed_keys = cat_utils.vet_main_table(mae_tbl)
    assert disallowed_keys[0] == 'LL_orig'

    # Write
    mae_tbl.to_parquet(outfile)


def mae_ulmo_evaluate(tbl_file:str, img_files:list,
                  clobber=False, debug=False, 
                  model='viirs-98'):
    """ Run Ulmo on MAE cutouts with the given model
    """
    
    # Load
    mae_table = ulmo_io.load_main_table(tbl_file)

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
        mae_table = ulmo_evaluate.eval_from_main(mae_table,
                                 model=model)

        # Save LL
        mae_table[f'LL_t{t_per}_p{p_per}'] = mae_table.LL

        # Could save after each eval..

    # Vet
    chk, disallowed_keys = cat_utils.vet_main_table(mae_table)
    for key in disallowed_keys:
        assert key[0:2] == 'LL'

    # Write 
    ulmo_io.write_main_table(mae_table, tbl_file)


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the MAE Table
    if flg & (2**0):
        gen_mae_tbl(llc_nonoise_file, mae_nonoise_file)

    # Evaluate LL with ulmo
    if flg & (2**1):

        # Ulmo model
        model='viirs-98'

        # Image parameters -- use str for the values
        img_pers = [('10', '10')]  

        # Generate the file names
        img_files = []
        for img_per in img_pers:
            base_name = f'mae_reconstruct_t{img_per[0]}_p{img_per[1]}.h5'
            img_file = os.path.join(mae_img_path, base_name)
            #
            img_files.append(img_file)
        
        # Run
        mae_ulmo_evaluate(mae_nonoise_file, img_files,
                          model=model, clobber=False)


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

# Setup
# python -u llc_uniform_144km.py 1

# Extract with noise
# python -u llc_uniform_144km.py 2 

# Evaluate -- run in Nautilus
# python -u llc_uniform_144km.py 4

# Evaluate without noise -- run in Nautilus
# python -u llc_uniform_144km.py 8