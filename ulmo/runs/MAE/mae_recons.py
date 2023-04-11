""" Module to perform reconstructions for MAE
"""
import os
import numpy as np

import h5py

from skimage.restoration import inpaint as sk_inpaint


from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate
from ulmo.utils import catalog as cat_utils
from ulmo.mae import mae_utils
from ulmo.modis import analysis as modis_analysis
from ulmo.mae import cutout_analysis

from IPython import embed

llc_tst_file = 's3://llc/Tables/test_uniform144_r0.5_test.parquet'
llc_full_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
llc_nonoise_file = 's3://llc/Tables/LLC_uniform144_r0.5_nonoise.parquet'

# MAE
mae_tst_nonoise_file = 's3://llc/mae/Tables/MAE_uniform144_test.parquet'
#mae_nonoise_file = 's3://llc/mae/Tables/MAE_uniform144_nonoise.parquet'
mae_valid_nonoise_tbl_file = 's3://llc/mae/Tables/MAE_LLC_valid_nonoise.parquet'
mae_valid_nonoise_file = 's3://llc/mae/PreProc/MAE_LLC_valid_nonoise_preproc.h5'
mae_img_path = 's3://llc/mae/PreProc'

ogcm_path = os.getenv('OS_OGCM')
if ogcm_path is not None:
    enki_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki')
    local_mae_valid_nonoise_file = os.path.join(enki_path, 'PreProc', 'MAE_LLC_valid_nonoise_preproc.h5')

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
            raise IOError("train should not be found")
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
    ulmo_io.write_main_table(viirs_100, viirs_100_s3_file)

def simple_inpaint(items):
    # Unpack
    img, mask = items
    return sk_inpaint.inpaint_biharmonic(img, mask, channel_axis=None)

def compare_with_inpainting(inpaint_file:str, t:int, p:int, debug:bool=False,
                            patch_sz:int=4, n_cores:int=10, 
                            nsub_files:int=5000,
                            local:bool=False):


    # Load images
    if local:
        local_recon_file = mae_utils.img_filename(t,p, local=True)
        local_mask_file = mae_utils.mask_filename(t,p, local=True)
        local_orig_file = local_mae_valid_nonoise_file
    else:
        recon_file = mae_utils.img_filename(t,p, local=False)
        mask_file = mae_utils.mask_filename(t,p, local=False)
        local_recon_file = os.path.basename(recon_file)
        local_mask_file = os.path.basename(mask_file)
        local_orig_file = os.path.basename(mae_valid_nonoise_file)
        # Download?
        for local_file, s3_file in zip(
            [local_recon_file, local_mask_file, local_orig_file],
            [recon_file, mask_file, mae_valid_nonoise_file]):
            if not os.path.exists(local_file):
                ulmo_io.download_file_from_s3(local_file, 
                                      s3_file)

    f_orig = h5py.File(local_orig_file, 'r')
    f_recon = h5py.File(local_recon_file,'r')
    f_mask = h5py.File(local_mask_file,'r')

    if debug:
        nfiles = 1000
        nsub_files = 100
        orig_imgs = f_orig['valid'][:nfiles,0,...]
        recon_imgs = f_recon['valid'][:nfiles,0,...]
        mask_imgs = f_mask['valid'][:nfiles,0,...]
    else:
        orig_imgs = f_orig['valid'][:,0,...]
        recon_imgs = f_recon['valid'][:,0,...]
        mask_imgs = f_mask['valid'][:,0,...]

    # Mask out edges
    mask_imgs[:, patch_sz:-patch_sz,patch_sz:-patch_sz] = 0

    # Analyze
    diff_recon = (orig_imgs - recon_imgs)*mask_imgs
    nfiles = diff_recon.shape[0]


    # Inpatinting
    map_fn = partial(simple_inpaint)

    nloop = nfiles // nsub_files + ((nfiles % nsub_files) > 0)
    inpainted = []
    for kk in range(nloop):
        i0 = kk*nsub_files
        i0 = kk*nsub_files
        i1 = min((kk+1)*nsub_files, nfiles)
        print('Files: {}:{} of {}'.format(i0, i1, nfiles))
        sub_files = [(diff_recon[ii,...], mask_imgs[ii,...]) for ii in range(i0, i1)]
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(
                sub_files) // n_cores if len(sub_files) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, sub_files,
                                                chunksize=chunksize), total=len(sub_files)))
        # Save
        inpainted.append(np.array(answers))
    # Collate
    inpainted = np.concatenate(inpainted)

    # Save
    with h5py.File(inpaint_file, 'w') as f:
        # Validation
        f.create_dataset('inpainted', data=inpainted.astype(np.float32))
    print(f'Wrote: {inpaint_file}')

def calc_rms(t:int, p:int, dataset:str='LLC', clobber:bool=False,
             debug:bool=False):


    if dataset == 'VIIRS':
        tbl_file = viirs_100_s3_file
        orig_file = viirs_100_img_file
        recon_file = os.path.join(sst_path, 'VIIRS', 'Enki', 'Recon',
                                  f'VIIRS_100clear_t{t}_p{p}.h5')
        mask_file = recon_file.replace('.h5', '_mask.h5')
    elif dataset == 'LLC':
        tbl_file = mae_valid_nonoise_tbl_file
        recon_file = mae_utils.img_filename(t,p, local=True)
        mask_file = mae_utils.mask_filename(t,p, local=True)
        orig_file = local_mae_valid_nonoise_file
    else:
        raise ValueError("Bad dataset")

    # Load table
    tbl = ulmo_io.load_main_table(tbl_file)

    # Already exist?
    RMS_metric = f'RMS_t{t}_p{p}'
    if RMS_metric in tbl.keys() and not clobber:
        print(f"RMS metric = {RMS_metric} already evaluated.  Skipping..")
        return

    # Open up
    f_orig = h5py.File(orig_file, 'r')
    f_recon = h5py.File(recon_file, 'r')
    f_mask = h5py.File(mask_file, 'r')

    # Do it!
    print("Calculating RMS metric")
    rms = cutout_analysis.rms_images(f_orig, f_recon, f_mask)

    # Add to table
    print("Adding to table")
    #if debug:
    #    embed(header='220 of mae_recons')
    if dataset == 'LLC':
        all_rms = np.nan * np.ones_like(tbl.LL_t35_p50)
        idx = np.isfinite(tbl.LL_t35_p50)
        all_rms[idx] = rms
    else:
        all_rms = rms
    tbl[RMS_metric] = all_rms
        
    # Vet
    chk, disallowed_keys = cat_utils.vet_main_table(
        tbl, return_disallowed=True, cut_prefix=['MODIS_'])
    for key in disallowed_keys:
        assert key[0:2] in ['LL','RM']

    # Write 
    if debug:
        embed(header='239 of mae_recons')
    else:
        ulmo_io.write_main_table(tbl, tbl_file)


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the VIIRS images
    if flg & (2**0):
        gen_viirs_images()#debug=True)

    # Generate the VIIRS images
    if flg & (2**1):
        compare_with_inpainting('LLC_inpaint_t10_p10.h5', 
                                10, 10, local=False)

    # Calculate RMS for various reconstructions
    if flg & (2**2):
        # VIIRS
        #calc_rms(10, 10, dataset='VIIRS')

        # LLC
        for t in [10,35,75]:
            for p in [10,20,30,40,50]:
                calc_rms(t, p, dataset='LLC')


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Images for VIIRS
        #flg += 2 ** 1  # 2 -- Inpaint vs Enki
        flg += 2 ** 2  # 4 -- RMS calculations
    else:
        flg = sys.argv[1]

    main(flg)

# Generate the VIIRS images
# python -u mae_recons.py 1

# Evaluate
# python -u mae_eval_ulmo.py 2
