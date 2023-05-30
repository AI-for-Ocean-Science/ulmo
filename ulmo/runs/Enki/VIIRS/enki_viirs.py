""" Module to enable VIIRS reconstructions
"""
import os
import numpy as np

import h5py
import pandas

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo.llc import io as llc_io
from ulmo.llc import extract 
from ulmo.llc import uniform

from ulmo.preproc import plotting as pp_plotting
from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils

from ulmo.mae import enki_utils
from ulmo.mae import cutout_analysis


from IPython import embed

llc_tst_file = 's3://llc/Tables/test_uniform144_r0.5_test.parquet'
llc_full_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
llc_nonoise_file = 's3://llc/Tables/LLC_uniform144_r0.5_nonoise.parquet'

llc_tot1km_tbl_file = 's3://llc/Tables/LLC_1km_r0.5.parquet'

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


def gen_llc_1km_table(tbl_file:str, debug:bool=False, 
                      resol:float=0.5, max_km:float=1.2, 
                      max_lat:float=None, plot:bool=True):
    """ Generate table for cutouts on ~1km scale 

    Args:
        tbl_file (str): Output name for Table. Should be in s3
        debug (bool, optional): _description_. Defaults to True.
        resol (float, optional): 
            Typical separation of images in deg
        max_lat (float, optional): Restrict on latitude
    """
    # Figure out lat range
    coords_ds = llc_io.load_coords()
    R_earth = 6371. # km
    circum = 2 * np.pi* R_earth
    km_deg = circum / 360.

    gd_lat = km_pix <= max_km

    # Begin
    llc_table = uniform.coords(
        resol=resol, max_lat=max_lat, min_lat=min_lat,
        field_size=(64,64), outfile=tbl_file)

    # Plot
    if plot:
        pp_plotting.plot_extraction(
            llc_table, s=1, resol=resol)

    # Temporal sampling
    if debug:
        # Extract 6 days across the full range;  ends of months
        dti = pandas.date_range('2011-09-13', periods=6, freq='2M')
    else:
        # Extract 52 days across the full range;  every 1 week
        dti = pandas.date_range('2011-09-13', periods=52, freq='1W')
    llc_table = extract.add_days(llc_table, dti, outfile=tbl_file)

    # Measure DT only
    llc_table = extract.preproc_for_analysis(
        llc_table, preproc_root='llc_std', dlocal=True,
        debug=debug)

    # Vet
    assert cat_utils.vet_main_table(llc_table)

    # Write 
    ulmo_io.write_main_table(llc_table, tbl_file)

    print(f"Wrote: {tbl_file} with {len(llc_table)} unique cutouts.")
    print("All done with init")


def balance_cutouts_log10DT(tot_tbl_file:str, 
                       balanced_tbl_file:str,
                       ncutouts:int,
                       nbins:int=10,
                       debug=False): 
    """ Generate a log10 DT balanced 

    Args:
        tot_tbl_file (str): _description_
        balanced_tbl_file (str): _description_
        ncutouts (int): _description_
        debug (bool, optional): _description_. Defaults to False.
    """
    # Giddy up (will take a bit of memory!)
    tot_tbl = ulmo_io.load_main_table(tot_tbl_file)

    # Steps
    log10DT = np.log10(tot_tbl.DT)
    logDT_steps = np.linspace(log10DT.min(),
                              log10DT.max(), 
                              nbins+1)
    nper_bin = ncutouts // nbins

    # Loop on bins                        
    save_idx = []
    for kk in range(nbins):
        # In bin
        in_bin = (log10DT > logDT_steps[kk]) & (
            log10DT <= logDT_steps[kk+1])
        idx = np.where(in_bin)[0]
        # Check
        assert len(idx) > nper_bin
        # Random choice
        save_idx.append(np.random.choice(
            idx, nper_bin, replace=False))
    save_idx = np.concatenate(save_idx)

    # 
    new_tbl = tot_tbl.iloc[save_idx]
    new_tbl.reset_index(inplace=True)

    # Vet
    assert cat_utils.vet_main_table(new_tbl)

    # Write 
    ulmo_io.write_main_table(new_tbl, balanced_tbl_file)



def extract_llc_cutouts( tbl_file:str, debug=False, 
                        debug_local=False, root_file=None, 
                        dlocal=True, preproc_root='llc_144', 
                        MAE=False):
    """Extract 64x64 cutouts 

    Args:
        tbl_file (str): filename for Table of cutouts
        debug (bool, optional): 
            Debug?
        debug_local (bool, optional): _description_. Defaults to False.
        root_file (_type_, optional): _description_. Defaults to None.
        dlocal (bool, optional): _description_. Defaults to False.
        preproc_root (str, optional): _description_. Defaults to 'llc_144'.
        dlocal (bool, optional): Use local files for LLC data.
    """

    # Giddy up (will take a bit of memory!)
    llc_table = ulmo_io.load_main_table(tbl_file)

    if debug:
        # Cut down to first 2 days
        uni_date = np.unique(llc_table.datetime)
        gd_date = llc_table.datetime <= uni_date[1]
        llc_table = llc_table[gd_date]
        debug_local = True

    if debug:
        root_file = 'MAE_LLC_uniform144_test_preproc.h5'
    else:
        if root_file is None:
            root_file = 'LLC_uniform144_preproc.h5'

    # Setup
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if MAE:
        pp_s3_file = pp_s3_file.replace('PreProc', 'mae/PreProc')
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    print(f"Outputting to: {pp_s3_file}")

    # Run it
    if debug_local:
        pp_s3_file = None  
    # Check indices
    assert np.all(np.arange(len(llc_table)) == llc_table.index)
    # Do it
    extract.preproc_for_analysis(llc_table, 
                                 pp_local_file,
                                 fixed_km=144.,
                                 preproc_root=preproc_root,
                                 s3_file=pp_s3_file,
                                 debug=debug,
                                 dlocal=dlocal,
                                 override_RAM=True)
    # Final write
    if not debug:
        ulmo_io.write_main_table(llc_table, tbl_file)
    print("You should probably remove the PreProc/ folder")


def inpaint(inpaint_file:str, 
            t:int, p:int, debug:bool=False,
            patch_sz:int=4, n_cores:int=10, 
            nsub_files:int=5000,
            local:bool=False):


    # Load images
    if local:
        local_recon_file = enki_utils.img_filename(t,p, local=True, dataset='VIIRS')
        local_mask_file = enki_utils.mask_filename(t,p, local=True, dataset='VIIRS')
        local_orig_file = viirs_100_img_file
    else:
        embed(header='Need to modify the following!')
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
    diff_recon = (orig_imgs - recon_imgs)*mask_imgs
    nfiles = diff_recon.shape[0]

    # Inpatinting
    map_fn = partial(cutout_analysis.simple_inpaint)

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
 

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the VIIRS images
    if flg & (2**0):
        gen_llc_1km_table(llc_tot1km_tbl_file)

    # Generate the VIIRS images
    if flg & (2**1):
        inpaint('Enki_VIIRS_inpaint_t10_p10.h5', 
                10, 10, debug=False, local=True) 



# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Generate the total table
        #flg += 2 ** 1  # 2 -- Inpaint 
    else:
        flg = sys.argv[1]

    main(flg)

# Inpaint
# python -u enki_viirs.py 2