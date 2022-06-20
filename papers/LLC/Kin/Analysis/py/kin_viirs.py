""" Kinematic analysis on VIIRS data """
import sys, os
import numpy as np

import pandas
import h5py

from ulmo import io as ulmo_io
from ulmo.llc import kinematics

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import kin_utils

def grab_image(cutout:pandas.core.series.Series, 
               close=True, pp_hf=None, use_local=False):
    """Grab a cutout image

    Args:
        cutout (pandas.core.series.Series): [description]
        close (bool, optional): [description]. Defaults to True.
        pp_hf ([type], optional): Pointer to the HDF5 file. Defaults to None.
        local_file (str, optional): Use this file, if provided

    Returns:
        np.ndarray or tuple: image or (image, hdf pointer)
    """
    if use_local:
        basefile = os.path.basename(cutout.pp_file)
        local_file = os.path.join(os.getenv('SST_OOD'), 'VIIRS', 'PreProc', basefile)
        pp_hf = h5py.File(local_file, 'r')
    # Open?
    if pp_hf is None:
        with ulmo_io.open(cutout.pp_file, 'rb') as f:
            pp_hf = h5py.File(f, 'r')
    if cutout.pp_type != 0:
        raise ValueError(f'{cutout.pp_type} not supported')
    img = pp_hf['valid'][cutout.pp_idx, 0, ...]

    # Close?
    if close:
        pp_hf.close()
        return img
    else:
        return img, pp_hf

def brazil_pdfs(outfile='viirs_brazil_kin_cutouts.npz', debug=False):
    """ Generate the Kin cutouts of F_S, w_z, Divb2 for DT ~ 1K cutouts
    in the Brazil-Malvanis confluence
    """
    # Load LLC
    viirs_tbl_file = 's3://viirs/Tables/VIIRS_all_98clear_std.parquet'
    viirs_table = ulmo_io.load_main_table(viirs_tbl_file)

    evals_bz, idx_R1, idx_R2 = kin_utils.grab_brazil_cutouts(
        viirs_table, dDT=0.25) # Higher dDT for stats


    # R1 first
    R1_F_s, R1_W, R1_divb, R1_divT = [], [], [], []
    for kk, iR1 in enumerate(idx_R1):
        if debug and kk > 1:
            continue
        print(f'R1: {kk} of {idx_R1.size}')
        cutout = evals_bz.iloc[iR1]
        # Load  -- These are done local
        SST = grab_image(cutout, close=True)
        # Calculate F_s
        divT = kinematics.calc_gradT(SST)
        # Store
        R1_divT.append(divT)

    # R2 first
    R2_F_s, R2_W, R2_divb, R2_divT = [], [], [], []
    for kk, iR2 in enumerate(idx_R2):
        if debug and kk > 1:
            continue
        print(f'R2: {kk} of {idx_R2.size}')
        cutout = evals_bz.iloc[iR2]
        # Load 
        SST = grab_image(cutout, close=True)
        # Calculate
        divT = kinematics.calc_gradT(SST)
        # 
        R2_divT.append(divT)

    # Output
    np.savez(outfile, 
             R1_divT=np.stack(R1_divT),
             R2_divT=np.stack(R2_divT),
             )
    print(f"Wrote: {outfile}")

# Command line execution
if __name__ == '__main__':
    brazil_pdfs()
