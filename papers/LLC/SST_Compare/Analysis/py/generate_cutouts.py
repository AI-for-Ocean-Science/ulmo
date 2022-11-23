""" Generate files of cutouts for PMC """
import healpy as hp
import h5py
import numpy as np
import os
import pandas

from ulmo import io as ulmo_io
from ulmo.utils import image_utils 

import sst_compare_utils

from IPython import embed


def grab_healpix_cutouts(dataset:str, nside=64, 
                       local_file='equatorial_cutouts.h5', 
                       lon=-120.,  # Define the equator 
                       lat = 0.,
                       local=False):

    # Load table
    eval_tbl = sst_compare_utils.load_table(dataset, local=local)

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

    write_cutouts(cut_tbl, local_file)

def grab_rectangular_cutouts(dataset:str, outfile:str, 
                             lon_minmax:tuple, lat_minmax:tuple, 
                             local:bool=False, debug:bool=False):
    # Load table
    eval_tbl = sst_compare_utils.load_table(dataset, local=local)

    # Cut on lon/lat
    gd_lon = (eval_tbl.lon > lon_minmax[0]) & (eval_tbl.lon <= lon_minmax[1])
    gd_lat = (eval_tbl.lat > lat_minmax[0]) & (eval_tbl.lat <= lat_minmax[1])

    cut_tbl = eval_tbl[gd_lon & gd_lat].copy()
    cut_tbl.reset_index(drop=True, inplace=True)
    if debug:
        embed(header='62 of generate_cutouts')

    # Write
    write_cutouts(cut_tbl, outfile, local=local)


def write_cutouts(cut_tbl:pandas.DataFrame, outfile:str,
                  debug=False, local:bool=False):

    images = []
    ii = 0
    for idx in cut_tbl.index:
        print(f'{ii} of {len(cut_tbl)}')
        cutout = cut_tbl.loc[idx]
        if local:
            if 'viirs' in cutout.pp_file:
                path = os.path.join(os.getenv('SST_OOD'), 'VIIRS')
            elif 'llc' in cutout.pp_file:
                path = os.path.join(os.getenv('SST_OOD'), 'LLC')
            else:
                raise ValueError("Not ready for this")
            lppfile = os.path.join(path, 'PreProc', 
                                      os.path.basename(cutout.pp_file))
        else:
            lppfile = None
        # Grab it
        img = image_utils.grab_image(cutout, local_file=lppfile)
        # Save it
        images.append(img)
        ii += 1

    # Write to disk
    clms = list(cut_tbl.keys())
    
    print("Writing: {}".format(outfile))
    with h5py.File(outfile, 'w') as f:
        # Validation
        f.create_dataset('valid', data=np.array(images).astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', 
                                data=cut_tbl.to_numpy(dtype=str).astype('S'))
        dset.attrs['columns'] = clms
    print("Wrote: {}".format(outfile))


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
    #grab_cutouts(lon=122.81, lat=-54.34, 
    #             local_file='ACC_cutouts_44514_VIIRS.h5')
    #grab_cutouts(lon=122.81, lat=-54.34, 
    #             local_file='ACC_cutouts_44514_LLC.h5',
    #             llc=True, local=True)

    # PMC ACC E -- Same 
    #grab_cutouts(lon=115.83333, lat=-49.70239,
    #             local_file='ACC_cutouts_43282_VIIRS.h5')
    #grab_cutouts(lon=115.83333, lat=-49.70239,
    #             local_file='ACC_cutouts_43282_LLC.h5',
    #             llc=True, local=True)

    # PMC ACC 43497 -- Same 
    #grab_cutouts(lat=-50.4800445, lon=116.32075,
    #             local_file='ACC_cutouts_43497_VIIRS.h5')
    #grab_cutouts(lat=-50.4800445, lon=116.32075,
    #             local_file='ACC_cutouts_43497_LLC.h5',
    #             llc=True, local=True)

    # #########################################
    # Rectangular cutouts

    '''
    # Equatorial
    grab_rectangular_cutouts('viirs', 'viirs_eq_rect_cutouts.h5',
                             (245.-360, 255.-360), (-2., 2.), local=True)#, debug=True)
    grab_rectangular_cutouts('llc_match', 'llc_eq_rect_cutouts.h5',
                             (245.-360, 255.-360), (-2., 2.), local=True)#, debug=True)
    '''

    '''
    # Gulf                            
    grab_rectangular_cutouts('viirs', 'viirs_gulf_rect_cutouts.h5',
                             (290.-360, 310.-360), (34., 42.), local=True)#, debug=True)
    '''
    grab_rectangular_cutouts('llc_match', 'llc_gulf_rect_cutouts.h5',
                             (290.-360, 310.-360), (34., 42.), local=True)#, debug=True)
