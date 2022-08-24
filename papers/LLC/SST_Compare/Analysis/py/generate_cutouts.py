""" Generate files of cutouts for PMC """
import healpy as hp
import h5py
import numpy as np

from ulmo import io as ulmo_io
from ulmo.utils import image_utils 

from IPython import embed

def equatorial_cutouts(nside=64, local_file='equatorial_cutouts.h5'):

    # Load table
    table_file = 's3://viirs/Tables/VIIRS_all_98clear_std.parquet'
    eval_tbl = ulmo_io.load_main_table(table_file)

    # Define the equator
    lon=-120.
    lat = 0.

    # Healpix coord
    theta = (90 - lat) * np.pi / 180.  # convert into radians
    phi = lon * np.pi / 180.
    idx = hp.pixelfunc.ang2pix(nside, theta, phi) 

    # Now grab them all
    lats = eval_tbl.lat.values
    lons = eval_tbl.lon.values

    # Grab LL values
    vals = eval_tbl.LL.values

    # Healpix coords
    theta = (90 - lats) * np.pi / 180.  # convert into radians
    phi = lons * np.pi / 180.
    idx_all = hp.pixelfunc.ang2pix(nside, theta, phi) # returns the healpix pixel numbers that correspond to theta and phi values

    # Match
    gd = idx_all == idx

    # Cutouts
    cut_tbl = eval_tbl[gd]

    images = []
    ii = 0
    for idx in cut_tbl.index:
        print(ii)
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
    equatorial_cutouts()

