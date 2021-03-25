""" Module for LLC analysis matched to MODIS"""
import os
import numpy as np

import pandas

from ulmo.models import io as model_io
from ulmo.llc import extract 
from ulmo.llc import uniform
from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate 

from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky

from IPython import embed

tbl_test_file = 's3://llc/Tables/test_modis_2019.feather'
modis_file = 's3://modis-l2/Tables/MODIS_L2_std.feather'

def modis_init_test(field_size=(64,64), CC_max=1e-4):
    """Build the main table for 1 year of MODIS (2019)
    """
    # Load MODIS
    modisl2_table = ulmo_io.load_main_table(modis_file)

    # Load up CC_mask
    CC_mask = llc_io.load_CC_mask(field_size=field_size, 
                                  local=False)
    # Cut
    good_CC = CC_mask.CC_mask.values < CC_max
    good_CC_idx = np.where(good_CC)

    # Build coords
    llc_lon = CC_mask.lon.values[good_CC].flatten()
    llc_lat = CC_mask.lat.values[good_CC].flatten()
    print("Building LLC SkyCoord")
    llc_coord = SkyCoord(llc_lon*units.deg + 180.*units.deg, 
                         llc_lat*units.deg, 
                         frame='galactic')
                                
    
    # Build total cutout matrix                                
    
    llc_table = uniform.coords(resol=resol, 
                               field_size=field_size, 
                               outfile=tbl_test_file)
    # Plot
    extract.plot_extraction(llc_table, s=1, resol=resol)

    # Extract 6 days across the full range;  ends of months
    dti = pandas.date_range('2011-09-01', periods=6, freq='2M')
    llc_table = extract.add_days(llc_table, dti, outfile=tbl_file)

    print("All done with init")


def u_extract(debug_local=False):

    # Giddy up (will take a bit of memory!)
    llc_table = ulmo_io.load_main_table(tbl_file)
    root_file = 'LLC_uniform_test_preproc.h5'
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    # Run it
    if debug_local:
        pp_s3_file = None  
    extract.preproc_for_analysis(llc_table, 
                                             pp_local_file,
                                             s3_file=pp_s3_file,
                                             dlocal=False)
    # Final write
    ulmo_io.write_main_table(llc_table, tbl_file)
    

def u_evaluate(clobber_local=False):
    
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Evaluate
    ulmo_evaluate.eval_from_main(llc_table)

    # Write 
    ulmo_io.write_main_table(llc_table, tbl_file)

def u_add_velocities():
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)
    
    # Velocities
    extract.velocity_stats(llc_table)

    # Write 
    ulmo_io.write_main_table(llc_table, tbl_file)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

        # MMT/MMIRS
    if flg & (2**0):
        u_init()

    if flg & (2**1):
        u_extract()

    if flg & (2**2):
        u_evaluate()

    if flg & (2**3):
        u_add_velocities()

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup coords
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate
        #flg += 2 ** 3  # 8 -- Velocities
    else:
        flg = sys.argv[1]

    main(flg)