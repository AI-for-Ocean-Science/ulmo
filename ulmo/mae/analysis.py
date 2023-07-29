""" Analysis code for Enki """

import numpy as np
import h5py

from ulmo.mae import bias as enki_bias
from ulmo.mae import enki_utils
from ulmo.mae import cutout_analysis
from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils

from IPython import embed


def calc_rms(t:int, p:int, dataset:str='LLC', clobber:bool=False,
            method:str=None, in_recon_file:str=None,
             debug:bool=False, remove_bias:bool=True,
             keys:list=None):
    """ Calculate the RMSE

    Args:
        t (int): train fraction
        p (int): patch fraction
        dataset (str, optional): Dataset. Defaults to 'LLC'.
        clobber (bool, optional): Clobber?
        remove_bias (bool, optional): Remove bias?
        debug (bool, optional): Debug?
        method (str, optional): Inpainting/analysis method. Defaults to None (aka Enki).
        in_recon_file (str, optional): Input recon file. Defaults to None.

    Raises:
        ValueError: _description_
    """
    print(f"Working on: t={t}, p={p} for {dataset}")
    tbl_file, orig_file, recon_file, mask_file = enki_utils.set_files(
        dataset, t, p)

    if in_recon_file is not None:
        recon_file = in_recon_file

    # Load table
    tbl = ulmo_io.load_main_table(tbl_file)

    if remove_bias:
        # Load
        bias_value = enki_utils.load_bias((t,p), dataset=dataset)
    else:
        bias_value = 0.


    # Already exist?
    if method is None:
        RMS_metric = f'RMS_t{t}_p{p}'
    else:
        RMS_metric = f'RMS_{method}_t{t}_p{p}'
    if RMS_metric in tbl.keys() and not clobber:
        print(f"RMS metric = {RMS_metric} already evaluated.  Skipping..")
        return

    # Open up
    print(f'Opening: {orig_file}')
    print(f'Opening: {recon_file}')
    print(f'Opening: {mask_file}')

    # Use noiseless patches?
    if method == 'noiseless':
        # Allow for various noise models
        noise_root = dataset.split('_')[1]
        # Do it
        orig_file = orig_file.replace(noise_root, 'nonoise')
        print(f'Now using original file: {orig_file}')

    f_orig = h5py.File(orig_file, 'r')
    f_recon = h5py.File(recon_file, 'r')
    f_mask = h5py.File(mask_file, 'r')

    # Do it!
    print("Calculating RMS metric")
    rms = cutout_analysis.rms_images(f_orig, f_recon, f_mask, debug=debug,
                                     bias_value=bias_value,
                                     keys=keys)

    # Check one (or more)
    if debug:
        tbl_idx = 354315 # High DT
        idx = tbl.iloc[tbl_idx].pp_idx 
        orig_img = f_orig['valid'][idx,0,...]
        recon_img = f_recon['valid'][idx,0,...]
        mask_img = f_mask['valid'][idx,0,...]
        irms = cutout_analysis.rms_single_img(orig_img, recon_img, mask_img,
                                              bias_value=bias_value)

    # Add to table
    print("Adding to table")
    if debug:
        embed(header='231 of mae_recons')
    if dataset[0:3] == 'LLC':
        # Allow for bad/missing images
        all_rms = np.nan * np.ones(len(tbl))
        pp_idx = tbl.pp_idx.values
        for ss in range(len(tbl)):
            if pp_idx[ss] >= 0:
                rms_val = rms[pp_idx[ss]]
                all_rms[ss] = rms_val
    else:
        all_rms = rms[tbl.pp_idx]

    # Finally
    tbl[RMS_metric] = all_rms
        
    # Vet
    chk, disallowed_keys = cat_utils.vet_main_table(
        tbl, return_disallowed=True, cut_prefix=['MODIS_'])
    for key in disallowed_keys:
        assert key[0:2] in ['LL','RM', 'DT']

    # Write 
    if debug:
        embed(header='239 of mae_recons')
    else:
        ulmo_io.write_main_table(tbl, tbl_file)

