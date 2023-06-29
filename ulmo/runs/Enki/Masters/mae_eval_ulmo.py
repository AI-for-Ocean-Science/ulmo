""" Module to run all analysis related to fixed 144km Uniform sampling of LLC 
 144km is equivalent to 64 pixels at VIIRS sampling binned by 3
"""
import os
import numpy as np

import h5py
import healpy

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate
from ulmo.utils import catalog as cat_utils
from ulmo.mae import mae_utils
from ulmo.modis import analysis as modis_analysis
from ulmo.mae import patch_analysis

from IPython import embed

llc_tst_file = 's3://llc/Tables/test_uniform144_r0.5_test.parquet'
llc_full_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
llc_nonoise_file = 's3://llc/Tables/LLC_uniform144_r0.5_nonoise.parquet'

# MAE
mae_tst_nonoise_file = 's3://llc/mae/Tables/MAE_uniform144_test.parquet'
#mae_nonoise_file = 's3://llc/mae/Tables/MAE_uniform144_nonoise.parquet'
mae_valid_nonoise_file = 's3://llc/mae/Tables/MAE_LLC_valid_nonoise.parquet'
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
        gd_tbl = (mae_table.pp_idx >= 0) & (
            mae_table.pp_idx < nimg)
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

def mae_modis_cloud_cover(outfile='modis_2020_cloudcover.npz',
    year:int=2020, nside:int=64, debug=False, local=True,
    n_cores=10,
    nsub_files=1000):

    
    CC_values = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 
                 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    #if debug:
    #    modis_analysis.cloud_cover_granule('/tank/xavier/Oceanography/data/MODIS/SST/night/2020/AQUA_MODIS.20200822T095001.L2.SST.nc',
    #        CC_values=CC_values, nside=nside)

    # Setup for preproc
    map_fn = partial(modis_analysis.cloud_cover_granule, 
                     CC_values=CC_values, nside=nside)

    # Healpix
    npix_hp = healpy.nside2npix(nside)
    all_pix_CC = np.zeros((npix_hp, len(CC_values)), dtype=int)
    tot_pix_CC = np.zeros(len(CC_values), dtype=int)


    files = []
    if local:
        local_path = os.path.join(os.getenv('MODIS_DATA'), 'night', f'{year}') 
        for root, dirs, ifiles in os.walk(os.path.abspath(local_path)):
            for ifile in ifiles:
                files.append(os.path.join(root,ifile))
    if debug:
        # Grab 100 random
        #files = shuffle(files, random_state=1234)
        ndebug_files=16
        files = files[:ndebug_files]  # 10%
        n_cores = 4
        #files = files[:100]

    nloop = len(files) // nsub_files + ((len(files) % nsub_files) > 0)
    bad_files = []
    for kk in range(nloop):
        #
        i0 = kk*nsub_files
        i1 = min((kk+1)*nsub_files, len(files))
        print('Files: {}:{} of {}'.format(i0, i1, len(files)))
        sub_files = files[i0:i1]

        # Download
        basefiles = []
        if not local:
            print("Downloading files from s3...")
        for ifile in sub_files:
            if local:
                basefiles = sub_files
            else:
                basename = os.path.basename(ifile)
                basefiles.append(basename)
                # Already here?
                if os.path.isfile(basename):
                    continue
                try:
                    ulmo_io.download_file_from_s3(basename, ifile, verbose=False)
                except:
                    print(f'Downloading {basename} failed')
                    bad_files.append(basename)
                    # Remove from sub_files
                    sub_files.remove(ifile)
                    continue
                
        if not local:
            print("All Done!")

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(sub_files) // n_cores if len(sub_files) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, basefiles,
                                            chunksize=chunksize), 
                                total=len(sub_files)))
        # Parse
        print("Parsing results...")
        for items in answers:
            if items is None:
                continue
            tot_pix, hp_idx = items
            # Parse
            for kk, idx in enumerate(hp_idx):
                all_pix_CC[idx, kk] += 1
                tot_pix_CC[kk] += tot_pix[kk]

    # Save
    np.savez(outfile, CC_values=CC_values, hp_pix_CC=all_pix_CC, 
             tot_pix_CC=tot_pix_CC, nside=nside)
    print(f"Wrote: {outfile}")

def mae_patch_analysis(img_files:list, n_cores,
                  clobber=False, debug=False,
                  p_sz=4): 
    """ Evaluate paches in the reconstruction outputs
    """
    stats=['meanT', 'stdT', 'DT', 'median_diff', 
           'std_diff', 'max_diff', 'i_patch', 'j_patch',
           'DT_recon']
    
    # Loop on reconstructed files
    for recon_file in img_files:
        #t_per, p_per = mae_utils.parse_mae_img_file(recon_file)
        patch_analysis.anlayze_full_test(recon_file, n_cores=n_cores,
                       stats=stats,
                       p_sz=p_sz,
                       debug=debug)



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
        debug = False

        # Image parameters -- (train_percenntage, patch_percentage)
        #img_pers = [(10, 10), (10,20)]  
        #img_pers = [(75, 10), (75, 20), (75, 30), 
        #            (75, 40), (75, 50)]
        #img_pers = [(10, 10), (10, 20), (10, 30), 
        #            (10, 40), (10, 50)]
        img_pers = [(35, 10), (35, 20), (35, 30), 
                    (35, 40), (35, 50)]

        # Generate the file names
        img_files = []
        for img_per in img_pers:
            img_file = mae_utils.img_filename(img_per[0], img_per[1])
            if debug:
                img_file = img_file.replace('.h5', '_small.h5')
            img_files.append(img_file)
        
        # Run
        mae_ulmo_evaluate(mae_valid_nonoise_file, 
                          img_files,
                          model=model, clobber=False, 
                          debug=debug)

    # Calcualte MODIS cloud cover
    if flg & (2**2):
        debug = False
        mae_modis_cloud_cover(debug=debug)

    # Patch analysis
    if flg & (2**3):

        # Ulmo model
        debug = False
        n_cores = 6

        # Image parameters -- (train_percenntage, patch_percentage)
        #img_pers = [(10, 20)]
        img_pers = [(75, 10), (75, 20), (75, 30), 
                    (75, 40), (75, 50)]

        # Generate the file names
        img_files = []
        for img_per in img_pers:
            img_file = mae_utils.img_filename(
                img_per[0], img_per[1])
            img_files.append(img_file)
        
        # Run
        mae_patch_analysis(
            img_files, clobber=False, debug=debug,
            n_cores=n_cores, p_sz=4)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup Table
        #flg += 2 ** 1  # 2 -- Evaluate 
        #flg += 2 ** 2  # 4 -- Cloud cover
        #flg += 2 ** 3  # 8 -- Patch analysis
    else:
        flg = sys.argv[1]

    main(flg)

# Generate the table(s)
# python -u mae_eval_ulmo.py 1

# Evaluate
# python -u mae_eval_ulmo.py 2
