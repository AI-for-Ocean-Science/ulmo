""" Module for LLC analysis matched to MODIS"""
import os
import numpy as np

import pandas

from ulmo.llc import extract 
from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.utils import catalog as cat_utils


from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky

from IPython import embed

tbl_test_file = 's3://llc/Tables/test_modis2012.parquet'
modis_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'

def modis_init_test(field_size=(64,64), CC_max=1e-4, show=False):
    """Build the main table for ~1 year of MODIS (2012)
    """
    # Load MODIS
    print("Loading MODIS...")
    modisl2_table = ulmo_io.load_main_table(modis_file)

    # Load up CC_mask
    CC_mask = llc_io.load_CC_mask(field_size=field_size, 
                                  local=True)
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
    del CC_mask                            
    
    # Times
    
    # LLC files
    llc_files = np.array(
        ulmo_io.list_of_bucket_files('llc', prefix='/ThetaUVSalt'))
    times = [os.path.basename(ifile)[8:-3].replace('_',':') for ifile in llc_files]
    llc_dti = pandas.to_datetime(times)

    # Final grid for extractions
    llc_grid = np.zeros((len(llc_coord), len(llc_dti)), dtype=bool)

    # Cuts
    y2012 = modisl2_table.pp_file == 's3://modis-l2/PreProc/MODIS_R2019_2012_95clear_128x128_preproc_std.h5'
    in_llc = modisl2_table.datetime < llc_dti.max()
    modis_2012 = modisl2_table.loc[y2012&in_llc].copy()

    # Coords
    modis_coord = SkyCoord(modis_2012.lon.values*units.deg + 180.*units.deg, 
                     modis_2012.lat.values*units.deg, 
                     frame='galactic')

    # Match up
    print("Matching on coords...")
    idx, sep2d, _ = match_coordinates_sky(modis_coord, llc_coord, 
                                          nthneighbor=1)

    # Match in time
    
    # Pivot around 2012
    modis_dt = modis_2012.datetime - pandas.Timestamp('2012-01-01')                                        
    hours = np.round(modis_dt.values / np.timedelta64(1, 'h')).astype(int)
    
    # Round to every 12 hours
    hour_12 = 12*np.round(hours / 12).astype(int)
    # LLC
    llc_hours = np.round((llc_dti-pandas.Timestamp('2012-01-01')).values / np.timedelta64(1, 'h')).astype(int)

    # Finally match!
    mt_t = cat_utils.match_ids(hour_12, llc_hours)#, require_in_match=False)

    # Indexing and Assigning
    assert mt_t.max() < 1000
    tot_idx = idx*1000 + mt_t
    complete = np.zeros(tot_idx.size, dtype=bool)

    # Unique
    uval, uidx = np.unique(tot_idx, return_index=True)

    # Update grid
    llc_grid[idx[uidx], mt_t[uidx]] = True
    complete[uidx] = True

    # HERE IS WHERE WE WOULD DEAL WITH DUPS

    # Table time
    modis_llc = modis_2012.iloc[uidx].copy()

    # Rename
    modis_llc = modis_llc.rename(columns=dict(lat='modis_lat', lon='modis_lon',
                                         row='modis_row', col='modis_col',
                                         datetime='modis_datetime'))

    # Fill in LLC
    modis_llc['lat'] = llc_lat[idx[uidx]]
    modis_llc['lon'] = llc_lon[idx[uidx]]
    s3_llc_files = np.array(['s3://llc/'+llc_file for llc_file in llc_files])
    modis_llc['pp_file'] = s3_llc_files[mt_t[uidx]]

    modis_llc['row'] = good_CC_idx[0][idx[uidx]] - field_size[0]//2 # Lower left corner
    modis_llc['col'] = good_CC_idx[1][idx[uidx]] - field_size[0]//2 # Lower left corner

    modis_llc['datetime'] = llc_dti[mt_t[uidx]]
    
    # Plot
    if show: 
        # Hide cartopy
        from ulmo.preproc import plotting as pp_plotting
        pp_plotting.plot_extraction(modis_llc, figsize=(9,6))

    # Vet
    assert cat_utils.vet_main_table(modis_llc, cut_prefix='modis_')

    # Write
    ulmo_io.write_main_table(modis_llc, tbl_test_file)

    print("All done with test init")


def modis_extract(test=True, debug_local=False):

    # Giddy up (will take a bit of memory!)
    if test:
        tbl_file = tbl_test_file
        root_file = 'LLC_modis2012_test_preproc.h5'
    else:
        raise IOError("Not ready for anything but testing..")
    llc_table = ulmo_io.load_main_table(tbl_file)
    # Rename MODIS columns
    llc_table = llc_table.rename(columns=dict(filename='modis_filename'))

    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    # Run it
    if debug_local:
        pp_s3_file = None  
    llc_table = extract.preproc_for_analysis(llc_table, 
                                 pp_local_file,
                                 s3_file=pp_s3_file,
                                 dlocal=False)
    # Vet
    assert cat_utils.vet_main_table(llc_table, cut_prefix='modis_')

    # Final write
    ulmo_io.write_main_table(llc_table, tbl_file)
    

def modis_evaluate(test=True, clobber_local=False):

    if test:
        tbl_file = tbl_test_file
    else:
        raise IOError("Not ready for anything but testing..")
    
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)


    # Evaluate
    ulmo_evaluate.eval_from_main(llc_table)

    # Write 
    assert cat_utils.vet_main_table(llc_table, cut_prefix='modis_')
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
        modis_init_test(show=True)

    if flg & (2**1):
        modis_extract()

    if flg & (2**2):
        modis_evaluate()

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