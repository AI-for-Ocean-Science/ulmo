""" Module to perform and analyze reconstructions for Enki
"""
import os
import numpy as np
from pkg_resources import resource_filename

import h5py
import pandas

from skimage.restoration import inpaint as sk_inpaint

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate
from ulmo.mae import enki_utils
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
        local_recon_file = enki_utils.img_filename(t,p, local=True)
        local_mask_file = enki_utils.mask_filename(t,p, local=True)
        local_orig_file = local_mae_valid_nonoise_file
    else:
        recon_file = enki_utils.img_filename(t,p, local=False)
        mask_file = enki_utils.mask_filename(t,p, local=False)
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
    raise ValueError("I think the line of code below is a bug..")
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


def calc_bias(dataset:str='LLC', clobber:bool=False, debug:bool=False,
              update:list=None):
    """ Calculate the bias

    Args:
        dataset (str, optional): Dataset. Defaults to 'LLC'.
        clobber (bool, optional): Clobber?
        debug (bool, optional): Debug?
        update (list, optional): 
            list of t,p values to update.  If None, return if outfile exists

    Raises:
        ValueError: _description_
    """
    outfile = f'enki_bias_{dataset}.csv'
    if os.path.isfile(outfile) and not clobber:
        if update is not None:
            df = pandas.read_csv(outfile)
        else:
            print(f"{outfile} already exists.  Returning..")
            return

    # Loop me
    ts, ps, medians, means = [], [], [], []
    all_ts = [10,20,35,50,75] if dataset == 'LLC2_nonoise' else [10,35,50,75]
    for t in all_ts:
        for p in [10,20,30,40,50]:
            if update is not None:
                if (t,p) not in update:
                    print(f"Skipping: t={t}, p={p}")
                    continue
            # 
            print(f"Working on: t={t}, p={p}")
            _, orig_file, recon_file, mask_file = enki_utils.set_files(dataset, t, p)

            # Open up
            f_orig = h5py.File(orig_file, 'r')
            f_recon = h5py.File(recon_file, 'r')
            f_mask = h5py.File(mask_file, 'r')

            if debug:
                nimgs = 1000
            else:
                nimgs = None

            # Do it!
            print("Calculating bias metric")
            median_bias, mean_bias = cutout_analysis.measure_bias(
                f_orig, f_recon, f_mask, debug=debug, nimgs=nimgs)
            #if debug:
            #    embed(header='307 of mae_recons')
            # Save
            ts.append(t)
            ps.append(p)
            medians.append(median_bias)
            means.append(mean_bias)
            # Update?
            if update is not None:
                idx = np.where((df.t == t) & (df.p == p))[0][0]
                df.loc[idx, 'median'] = median_bias
                df.loc[idx, 'mean'] = mean_bias

    # Write
    if update is None:
        df = pandas.DataFrame(dict(t=ts, p=ps, median=medians, mean=means))
    df.to_csv(outfile, index=False)
    print(f"Wrote: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the VIIRS images
    if flg & (2**0):
        gen_viirs_images()#debug=True)

    # Inpainting 
    if flg & (2**1):
        compare_with_inpainting('LLC_inpaint_t10_p10.h5', 
                                10, 10, local=False)

    # Calculate RMS for various reconstructions
    if flg & (2**2):
        clobber = True
        debug=False
        # VIIRS
        #calc_rms(10, 10, dataset='VIIRS', clobber=clobber)

        # LLC
        #for t in [10,35,50,75]:
        #    for p in [10,20,30,40,50]:
        #for t in [35,75]:
        for t in [50]:
            for p in [10,20,30,40,50]:
                print(f'Working on: t={t}, p={p}')
                calc_rms(t, p, dataset='LLC', clobber=clobber, debug=debug)

    # Calculate RMS for various reconstructions
    if flg & (2**3):
        debug = False
        # VIIRS
        #calc_rms(10, 10, dataset='VIIRS', clobber=clobber)

        # LLC
        #calc_bias(dataset='LLC', debug=debug, clobber=True)
        #calc_bias(dataset='LLC', debug=debug,
        #          update=[(20,10),(20,20),(20,30),(20,40),(20,50)])

        calc_bias(dataset='LLC2_nonoise', debug=debug, clobber=True)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Images for VIIRS
        #flg += 2 ** 1  # 2 -- Inpaint vs Enki
        #flg += 2 ** 2  # 4 -- RMSE calculations
        #flg += 2 ** 3  # 8 -- bias calculations
    else:
        flg = sys.argv[1]

    main(flg)

# RMSE
# python -u enki_recons.py 4
