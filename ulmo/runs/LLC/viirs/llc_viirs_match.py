""" Module for LLC analysis matched to VIIRS """
import os
import numpy as np

import pandas
import datetime

from ulmo.llc import extract 
from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.utils import catalog as cat_utils

from ulmo.viirs import utils as viirs_utils


from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky

from IPython import embed

viirs_match_file = 's3://llc/Tables/llc_viirs_match.parquet'
tbl_test_noise_file = 's3://llc/Tables/test_noise_modis2012.parquet'
local_modis_file = '/home/xavier/Projects/Oceanography/AI/OOD/MODIS_L2/Tables/MODIS_L2_std.parquet'


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
    print("Loading VIIRS...")
    if localV:
        local_viirs_file = os.path.join(os.getenv('SST_OOD'),
                                'VIIRS', 'Tables',
                                'VIIRS_all_98clear_std.parquet')
        viirs_tbl = ulmo_io.load_main_table(local_viirs_file)
    else:
        embed(header='46 of viirs match table')
        modisl2_table = ulmo_io.load_main_table(modis_file)

    # Load up CC_mask for the coordinates
    embed(header='consider using a 128 field_size or however big it needs to be')
    CC_mask = llc_io.load_CC_mask(field_size=field_size, 
                                  local=localCC)
    # Cut
    good_CC = CC_mask.CC_mask.values < CC_max
    good_CC_idx = np.where(good_CC)

    # Build coords
    llc_lon = CC_mask.lon.values[good_CC].flatten()
    llc_lat = CC_mask.lat.values[good_CC].flatten()

    del CC_mask                            
    
    # Times
    
    # LLC files
    llc_files = ulmo_io.list_of_bucket_files('llc', prefix='/ThetaUVSalt')
    llc_files = [item for item in llc_files if ('T12_00' in item) or ('T00_00' in item)]
    times = [os.path.basename(ifile)[8:-3].replace('_',':') for ifile in llc_files]
    llc_dti = pandas.to_datetime(times)

    # Cut down to have 1 year of coverage and only 1 
    keep_llc = llc_dti > datetime.datetime(2011, 11, 16, 13)
    times = np.array(times)[keep_llc].tolist()
    llc_dti = pandas.to_datetime(times)
    llc_files = np.array(llc_files)[keep_llc]

    # LLC 12hr steps
    llc_start_year = pandas.to_datetime(llc_dti.year, format='%Y')
    llc_12hrs = (llc_dti - llc_start_year).total_seconds().astype(int) // (12*3600)
    uni_12hrs = np.unique(llc_12hrs)
    # These ought to all be unique
    assert llc_12hrs.size == uni_12hrs.size

    # VIIRS
    viirs_dti = pandas.to_datetime(viirs_tbl.datetime.values)
    viirs_start_year = pandas.to_datetime(viirs_dti.year, format='%Y')
    viirs_12hrs = (viirs_dti - viirs_start_year).total_seconds().astype(int) // (12*3600)

    # Match
    mt_t = cat_utils.match_ids(viirs_12hrs, llc_12hrs, require_in_match=False)
    keep = mt_t >= 0

    # Cut and do it again
    viirs_llc_tbl = viirs_tbl[keep].copy()
    mt_t = cat_utils.match_ids(viirs_12hrs[keep], llc_12hrs) 

    print("Building LLC SkyCoord")
    llc_coord = SkyCoord(llc_lon*units.deg + 180.*units.deg, 
                         llc_lat*units.deg, 
                         frame='galactic')

    # Coords
    viirs_coord = SkyCoord(viirs_llc_tbl.lon.values*units.deg + 180.*units.deg, 
                     viirs_llc_tbl.lat.values*units.deg, 
                     frame='galactic')

    # Match up
    print("Matching on coords...")
    idx, sep2d, _ = match_coordinates_sky(
        viirs_coord, llc_coord, nthneighbor=1)

    print(f"Maximum separaton is: {np.max(sep2d).to('arcsec')}")


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
    assert cat_utils.vet_main_table(llc_viirs_tbl, cut_prefix=['viirs_', 'MODIS_'])

    # Write
    outfile = viirs_match_file

    ulmo_io.write_main_table(llc_viirs_tbl, outfile)
    print(f"All done generating the Table: {outfile}")


# EXTRACTION
def llc_viirs_extract(tbl_file:str, 
                      root_file=None, dlocal=True, 
                      preproc_root='llc_144', 
                      debug=False):

    # Giddy up (will take a bit of memory!)
    llc_table = ulmo_io.load_main_table(tbl_file)

    if debug:
        # Cut down to first 2 days
        uni_date = np.unique(llc_table.datetime)
        gd_date = llc_table.datetime <= uni_date[1]
        llc_table = llc_table[gd_date]
        debug_local = True

    if debug:
        root_file = 'LLC_VIIRS144_test_preproc.h5'
    else:
        if root_file is None:
            root_file = 'LLC_VIIRS144_preproc.h5'

    # Setup
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    print(f"Outputting to: {pp_s3_file}")

    # Run it
    #if debug_local:
    #    pp_s3_file = None  
    # Check indices
    llc_table.reset_index(drop=True, inplace=True)
    assert np.all(np.arange(len(llc_table)) == llc_table.index)
    # Do it
    if debug:
        embed(header='210 of llc viirs')
    extract.preproc_for_analysis(llc_table, 
                                 pp_local_file,
                                 fixed_km=144.,
                                 preproc_root=preproc_root,
                                 s3_file=pp_s3_file,
                                 #debug=debug,
                                 dlocal=dlocal,
                                 override_RAM=True)
    # Vet
    assert cat_utils.vet_main_table(llc_table, cut_prefix=['viirs_', 'MODIS_'])

    # Final write
    if not debug:
        ulmo_io.write_main_table(llc_table, tbl_file)
    print("You should probably remove the PreProc/ folder")
    
    
def llc_viirs_evaluate_144(tbl_file:str, 
                   debug=False,
                   model='viirs-98'):
    """ Run Ulmo on the cutouts with the given model
    """
    
    
    if debug:
        tbl_file = tst_file
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Evaluate
    ulmo_evaluate.eval_from_main(llc_table,
                                 model=model)

    # Vet
    assert cat_utils.vet_main_table(llc_table, cut_prefix=['viirs_', 'MODIS_'])

    # Write 
    ulmo_io.write_main_table(llc_table, tbl_file)

def viirs_add_uid(debug=False, clobber=True):
    """ Add a UID to the VIIRS table
    """
    # Load
    s3_path = 's3://viirs/Tables/'
    tbl_files = ulmo_io.list_of_bucket_files(
        s3_path, include_prefix=True) 
    if debug:
        tbl_files = tbl_files[0:1]

    for tbl_file in tbl_files:
        viirs_table = ulmo_io.load_main_table(tbl_file)

        if 'UID' in viirs_table.keys() and not clobber:
            print("UID already in table")
            continue

        # Add
        viirs_table['UID'] = viirs_utils.viirs_uid(viirs_table)

        # Vet
        assert cat_utils.vet_main_table(viirs_table,
                                        cut_prefix=['MODIS_'])

        # Write
        if not debug:
            ulmo_io.write_main_table(viirs_table, tbl_file)
        else:
            print("Not writing file")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Start us off
    if flg & (2**0):
        viirs_match_table(show=True)

    if flg & (2**1):
        llc_viirs_extract(viirs_match_file, debug=False)

    if flg & (2**2):
        llc_viirs_evaluate_144(viirs_match_file)

    # Add UID for *all* VIIRS tables
    if flg & (2**3):
        viirs_add_uid()#debug=True)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup coords and table
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate
        flg += 2 ** 3  # 8 -- UID

    else:
        flg = sys.argv[1]

    main(flg)
