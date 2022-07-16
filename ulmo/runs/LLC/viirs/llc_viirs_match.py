""" Module for LLC analysis matched to VIIRS """
import os
import numpy as np

import pandas
import xarray

import datetime

from ulmo.llc import extract 
from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.utils import catalog as cat_utils


from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky

from IPython import embed

viirs_match_file = 's3://llc/Tables/llc_viirs_match.parquet'
tbl_test_noise_file = 's3://llc/Tables/test_noise_modis2012.parquet'
modis_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
local_modis_file = '/home/xavier/Projects/Oceanography/AI/OOD/MODIS_L2/Tables/MODIS_L2_std.parquet'
local_viirs_file = os.path.join(os.getenv('SSD_OOD'),
                                'VIIRS', 'Tables',
                                'viirs_98_all.parquet')


def viirs_match_table(field_size=(64,64), CC_max=1e-6, 
                      show=False,
                    noise=False, localV=True, localCC=True):
    """Build the main table for a match of LLC to VIIRS

    Args:
        field_size (tuple, optional): [description]. Defaults to (64,64).
        CC_max ([type], optional): [description]. Defaults to 1e-4.
        show (bool, optional): [description]. Defaults to False.
        noise (bool, optional): [description]. Defaults to False.
        localM (bool, optional): Load MODIS from local disk. Defaults to False.
    """
    # Load MODIS
    print("Loading MODIS...")
    if localV:
        viirs_tbl = ulmo_io.load_main_table(local_viirs_file)
    else:
        embed(header='46 of viirs match table')
        modisl2_table = ulmo_io.load_main_table(modis_file)

    # Load up CC_mask for the coordinates
    CC_mask = llc_io.load_CC_mask(field_size=field_size, 
                                  local=localCC)
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

    # LLC 12hr steps
    llc_start_year = pandas.to_datetime(llc_dti.year, format='%Y')
    llc_12hrs = (llc_dti - llc_start_year).total_seconds().astype(int) // (12*3600)
    uni_12hrs = np.unique(llc_12hrs)
    # These ought to all be unique
    assert llc_12hrs.size == uni_12hrs.size

    # Cuts -- Could restrict to night-time here
    viirs_dti = pandas.to_datetime(viirs_tbl.datetime.values)
    viirs_start_year = pandas.to_datetime(viirs_dti.year, format='%Y')
    viirs_12hrs = (viirs_dti - viirs_start_year).total_seconds().astype(int) // (12*3600)

    # Match
    mt_t = cat_utils.match_ids(viirs_12hrs, llc_12hrs, require_in_match=False)
    keep = mt_t >= 0

    # Cut and do it again
    viirs_llc_tbl = viirs_tbl[keep].copy()
    mt_t = cat_utils.match_ids(viirs_12hrs[keep], llc_12hrs) 

    # Coords
    viirs_coord = SkyCoord(viirs_llc_tbl.lon.values*units.deg + 180.*units.deg, 
                     viirs_llc_tbl.lat.values*units.deg, 
                     frame='galactic')

    # Match up
    print("Matching on coords...")
    idx, sep2d, _ = match_coordinates_sky(
        viirs_coord, llc_coord, nthneighbor=1)

    print(f"Maximum separaton is: {np.max(sep2d).to('arcsec')}")

    # Match in time
    
    '''
    # Pivot around every 12 hrs
    modis_dt = modis_2012.datetime - pandas.Timestamp('2012-01-01')                                        
    hours = np.round(modis_dt.values / np.timedelta64(1, 'h')).astype(int)
    
    # Round to every 12 hours
    hour_12 = 12*np.round(hours / 12).astype(int)
    # LLC
    llc_hours = np.round((llc_dti-pandas.Timestamp('2012-01-01')).values / np.timedelta64(1, 'h')).astype(int)

    # Finally match!
    mt_t = cat_utils.match_ids(hour_12, llc_hours)#, require_in_match=False)
    '''

    # Indexing and Assigning
    assert mt_t.max() < 1000
    tot_idx = idx*1000 + mt_t
    complete = np.zeros(tot_idx.size, dtype=bool)

    # Unique
    uval, uidx = np.unique(tot_idx, return_index=True)

    if uval.size < mt_t.size:
        print("WARNING: We have duplicates!!!")

    
    # Final grid for extractions
    #  This requires ~100Gb of RAM
    llc_grid = np.zeros((len(llc_coord), len(llc_dti)), dtype=bool)

    llc_grid[idx[uidx], mt_t[uidx]] = True
    complete[uidx] = True

    # HERE IS WHERE WE WOULD DEAL WITH DUPS

    # Table time
    llc_viirs_tbl = viirs_llc_tbl.iloc[uidx].copy()

    # Rename
    llc_viirs_tbl = llc_viirs_tbl.rename(
        columns=dict(lat='viirs_lat', lon='viirs_lon', 
                     row='viirs_row', col='viirs_col',
                     datetime='viirs_datetime',
                     filename='viirs_filename',
                     UID='viirs_UID', LL='viirs_LL'))

    # Fill in LLC
    llc_viirs_tbl['lat'] = llc_lat[idx[uidx]]
    llc_viirs_tbl['lon'] = llc_lon[idx[uidx]]
    s3_llc_files = np.array(['s3://llc/'+llc_file for llc_file in llc_files])
    llc_viirs_tbl['pp_file'] = s3_llc_files[mt_t[uidx]]

    llc_viirs_tbl['row'] = good_CC_idx[0][idx[uidx]] - field_size[0]//2 # Lower left corner
    llc_viirs_tbl['col'] = good_CC_idx[1][idx[uidx]] - field_size[0]//2 # Lower left corner

    llc_viirs_tbl['datetime'] = llc_dti[mt_t[uidx]]
    
    # Plot
    if show: 
        # Hide cartopy
        from ulmo.preproc import plotting as pp_plotting
        pp_plotting.plot_extraction(llc_viirs_tbl, figsize=(9,6))

    # Vet
    assert cat_utils.vet_main_table(llc_viirs_tbl, cut_prefix='viirs')

    # Write
    outfile = viirs_match_file

    ulmo_io.write_main_table(llc_viirs_tbl, outfile)
    print("All done with test init.")


def modis_extract(test=True, debug_local=False, 
                  noise=False, debug=False):

    # Giddy up (will take a bit of memory!)
    if noise:
        preproc_root='llc_noise' 
    else:
        preproc_root='llc_std' 
    if test:
        if noise:
            tbl_file = tbl_test_noise_file
            root_file = 'LLC_modis2012_test_noise_preproc.h5'
        else:
            tbl_file = tbl_test_file
            root_file = 'LLC_modis2012_test_preproc.h5'
    else:
        raise IOError("Not ready for anything but testing..")
    llc_table = ulmo_io.load_main_table(tbl_file)
    # Rename MODIS columns
    if 'filename' in llc_table.keys() and 'modis_filename' not in llc_table.keys():
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
                                 preproc_root=preproc_root,
                                 s3_file=pp_s3_file,
                                 dlocal=False,
                                 debug=debug)
    # Vet
    assert cat_utils.vet_main_table(llc_table, cut_prefix='modis_')

    # Final write
    if not debug:
        ulmo_io.write_main_table(llc_table, tbl_file)
    

def modis_evaluate(test=True, noise=False, tbl_file=None, rename=True):

    if tbl_file is None:
        if test:
            tbl_file = tbl_test_noise_file if noise else tbl_test_file
        else:
            raise IOError("Not ready for anything but testing..")
    
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Rename
    if rename and 'LL' in llc_table.keys() and 'modis_LL' not in llc_table.keys():
        llc_table = llc_table.rename(
            columns=dict(LL='modis_LL'))

    # Evaluate
    llc_table = ulmo_evaluate.eval_from_main(llc_table)

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

    # 2012 + noise
    if flg & (2**3):
        modis_init_test(show=False, noise=True, localCC=False)
        #modis_init_test(show=True, noise=True, localCC=True)#, localM=False)

    if flg & (2**4):
        modis_extract(noise=True, debug=False)

    if flg & (2**5):
        modis_evaluate(noise=True)

    if flg & (2**6):  # Debuggin
        modis_evaluate(tbl_file='s3://llc/Tables/test2_modis2012.parquet')

    if flg & (2**7):  
        modis_evaluate(tbl_file='s3://llc/Tables/ulmo2_test.parquet')
    
    if flg & (2**8):
        modis_evaluate(tbl_file='s3://llc/Tables/LLC_modis_noise2.parquet')

    if flg & (2**9): 
        modis_evaluate(tbl_file='s3://llc/Tables/LLC_modis_noise_track.parquet')
    
    if flg & (2**10):
        modis_evaluate(tbl_file='s3://llc/Tables/LLC_uniform_test.parquet')

    if flg & (2**11): # 2048
        modis_evaluate(tbl_file='s3://llc/Tables/LLC_uniform_viirs_noise.parquet')

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup coords
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate
        #flg += 2 ** 3  # 8 -- Init test + noise
        #flg += 2 ** 4  # 16 -- Extract + noise
        #flg += 2 ** 5  # 32 -- Evaluate + noise
        #flg += 2 ** 6  # 64 -- Evaluate debug run
        #flg += 2 ** 7  # 128 -- Katharina's first noise try
        #flg += 2 ** 8  #256 -- Katharina: modis noise avgd
        #flg += 2 ** 9  #512 -- Katharina: modis along track noise
        #flg += 2 ** 10 #1024 -- Katharina: LLc uniformly sampled, no noise (not done)
        flg += 2 ** 11  #2048 -- Katharina: LLC uniform viirs along scan noise




    else:
        flg = sys.argv[1]

    main(flg)
