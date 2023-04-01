""" Module to perform reconstructions for MAE
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

# VIIRS
sst_path = os.getenv('OS_SST')
if sst_path is not None:
    viirs_file = os.path.join(sst_path, 'VIIRS', 'Tables', 'VIIRS_all_98clear_std.parquet')
    viirs_100_file = os.path.join(sst_path, 'VIIRS', 'Tables', 'VIIRS_all_100clear_std.parquet')
    viirs_100_s3_file = os.path.join('s3://viirs', 'Tables', 'VIIRS_all_100clear_std.parquet')
    viirs_100_img_file = os.path.join(sst_path, 'VIIRS', 'PreProc', 'VIIRS_all_100clear_preproc.h5')


def gen_viirs_images(debug:bool=False):
    """ Generate a file of cloud free VIIRS images
    """
    # Load
    viirs = ulmo_io.load_main_table(viirs_file)

    # Cut on CC
    all_clear = np.isclose(viirs.clear_fraction, 0.)
    viirs_100 = viirs[all_clear].copy()


    # Generate images
    uni_pps = np.unique(viirs_100.pp_file)
    if debug:
        uni_pps=uni_pps[0:2]

    the_images = []
    for pp_file in uni_pps:
        print(f'Working on {os.path.basename(pp_file)}')
        # Go local
        local_file = os.path.join(sst_path, 'VIIRS', 'PreProc', 
                                  os.path.basename(pp_file))
        f = h5py.File(local_file, 'r')
        if 'train' in f.keys():
            embed(header='55 of mae_recons')
        # Load em all (faster)
        data = f['valid'][:]
        in_file = viirs_100.pp_file == pp_file
        idx = viirs_100.pp_idx[in_file].values

        data = data[idx,...]
        the_images.append(data)

    # Write
    the_images = np.concatenate(the_images)
    with h5py.File(viirs_100_img_file, 'w') as f:
        # Validation
        f.create_dataset('valid', data=the_images.astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', 
                                data=viirs_100.to_numpy(dtype=str).astype('S'))
        #dset.attrs['columns'] = clms
        '''
        # Train
        if valid_fraction < 1:
            f.create_dataset('train', data=pp_fields[train_idx].astype(np.float32))
            dset = f.create_dataset('train_metadata', data=main_tbl.iloc[train_idx].to_numpy(dtype=str).astype('S'))
            dset.attrs['columns'] = clms
        '''
    print("Wrote: {}".format(viirs_100_img_file))

    # Write
    embed(header='80 of mae_recons')
    ulmo_io.write_main_table(viirs_100, viirs_100_s3_file)


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

    # Generate the VIIRS images
    if flg & (2**0):
        gen_viirs_images()#debug=True)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Images for VIIRS
    else:
        flg = sys.argv[1]

    main(flg)

# Generate the VIIRS images
# python -u mae_recons.py 1

# Evaluate
# python -u mae_eval_ulmo.py 2
