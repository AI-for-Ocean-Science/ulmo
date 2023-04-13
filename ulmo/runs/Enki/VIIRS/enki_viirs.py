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

def balance_cutouts_DT(tbl_file:str, debug=False): 
    llc_table = ulmo_io.load_main_table(tbl_file)

def extract_llc_cutouts( tbl_file:str, debug=False, 
                        debug_local=False, root_file=None, 
                        dlocal=True, preproc_root='llc_144', 
                        MAE=False):
    """Extract 64x64 cutouts 

    Args:
        tbl_file (str): Table of cutouts
        debug (bool, optional): _description_. Defaults to False.
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

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Images for VIIRS
        #flg += 2 ** 1  # 2 -- Inpaint vs Enki
    else:
        flg = sys.argv[1]

    main(flg)

# Generate the VIIRS images
# python -u mae_recons.py 1

# Evaluate
# python -u mae_eval_ulmo.py 2
