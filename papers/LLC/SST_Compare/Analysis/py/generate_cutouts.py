""" Generate files of cutouts for PMC """
import healpy as hp
import h5py
import numpy as np
import os

from ulmo import io as ulmo_io
from ulmo.utils import image_utils 

from IPython import embed

def grab_cutouts(nside=64, 
                       local_file='equatorial_cutouts.h5', 
                       llc=False,
                       lon=-120.,  # Define the equator 
                       lat = 0.,
                       local=False):

    # Load table
    if not llc:
        table_file = 's3://viirs/Tables/VIIRS_all_98clear_std.parquet'
    else:
        if local:
            table_file = os.path.join(
                os.getenv('SST_OOD'), 'LLC/Tables/llc_viirs_match.parquet')
        else:
            table_file = 's3://llc/Tables/llc_viirs_match.parquet'
    eval_tbl = ulmo_io.load_main_table(table_file)


    # Healpix coord
    theta = (90 - lat) * np.pi / 180.  # convert into radians
    phi = lon * np.pi / 180.
    idx = hp.pixelfunc.ang2pix(nside, theta, phi) 
    print(f"Healpix: {idx} at lon={lon}, lat={lat}")

    # Now grab them all
    lats = eval_tbl.lat.values
    lons = eval_tbl.lon.values

    # Healpix coords
    theta = (90 - lats) * np.pi / 180.  # convert into radians
    phi = lons * np.pi / 180.
    idx_all = hp.pixelfunc.ang2pix(nside, theta, phi) # returns the healpix pixel numbers that correspond to theta and phi values

    # Match
    gd = idx_all == idx
    print(f'There are {np.sum(gd)} cutouts')

    # Cutouts
    cut_tbl = eval_tbl[gd]

    images = []
    ii = 0
    for idx in cut_tbl.index:
        print(f'{ii} of {len(cut_tbl)}')
        cutout = cut_tbl.loc[idx]
        # Grab it
        img = image_utils.grab_image(cutout)
        # Save it
        images.append(img)
        ii += 1

    # Write to disk
    clms = list(cut_tbl.keys())

    
    print("Writing: {}".format(local_file))
    with h5py.File(local_file, 'w') as f:
        # Validation
        f.create_dataset('valid', data=np.array(images).astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', 
                                data=cut_tbl.to_numpy(dtype=str).astype('S'))
        dset.attrs['columns'] = clms
    print("Wrote: {}".format(local_file))


# Command line execution
if __name__ == '__main__':
    # Mine
    #grab_cutouts()

    # PMC Equitorial -- VIIRS
    #grab_cutouts(lon=-112.5, lat=0.5, local_file='equatorial_cutouts_pmc.h5')

    # PMC Equitorial -- LLC
    #grab_cutouts(lon=112.5, lat=0.5, local_file='equatorial_cutouts_pmc_llc.h5',
    #                   llc=True, local=True)

    # PMC ACC A 
    #grab_cutouts(lon=120.31, lat=-53.57, local_file='ACC_cutouts_44318_VIIRS.h5')
    #grab_cutouts(lon=120.31, lat=-53.57, local_file='ACC_cutouts_44318_LLC.h5',
    #             llc=True, local=True)

    # PMC ACC B 
    #grab_cutouts(lon=122.14, lat=-53.57, 
    #             local_file='ACC_cutouts_44319_VIIRS.h5')
    #grab_cutouts(lon=122.14, lat=-53.57, 
    #             local_file='ACC_cutouts_44319_LLC.h5',
    #             llc=True, local=True)

    # PMC ACC C 
    #grab_cutouts(lon=120.94, lat=-54.34, 
    #             local_file='ACC_cutouts_44513_VIIRS.h5')
    #grab_cutouts(lon=120.94, lat=-54.34, 
    #             local_file='ACC_cutouts_44513_LLC.h5',
    #             llc=True, local=True)

    # PMC ACC D 
    grab_cutouts(lon=122.81, lat=-54.34, 
                 local_file='ACC_cutouts_44514_VIIRS.h5')
    grab_cutouts(lon=122.81, lat=-54.34, 
                 local_file='ACC_cutouts_44514_LLC.h5',
                 llc=True, local=True)