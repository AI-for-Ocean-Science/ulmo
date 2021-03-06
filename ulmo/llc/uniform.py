""" Module for uniform analyses of LLC outputs"""

import os
import glob
import numpy as np

import pandas

import xarray as xr
import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

from ulmo.preproc import io as pp_io
from ulmo.preproc import utils as pp_utils

# Astronomy tools
import astropy_healpix
from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky

from sklearn.utils import shuffle

from IPython import embed

def setup_coords(resol):
    # Load up full grid of row, col
    allex_file = os.path.join(os.getenv('SST_OOD'), 'LLC', 
                           'Extractions', 'all_extract_coords_LLC.csv')
    df = pandas.read_csv(allex_file, index_col=0)

    # Healpix time
    nside = astropy_healpix.pixel_resolution_to_nside(resol*units.deg)
    hp = astropy_healpix.HEALPix(nside=nside)
    hp_lon, hp_lat = hp.healpix_to_lonlat(np.arange(hp.npix))

    # Coords
    hp_coord = SkyCoord(hp_lon, hp_lat, frame='galactic')
    llc_coord = SkyCoord(df.longitude.values*units.deg + 180.*units.deg, 
                         df.latitude.values*units.deg, frame='galactic')
                        
    # Cross-match
    idx, sep2d, _ = match_coordinates_sky(hp_coord, llc_coord, nthneighbor=1)
    flag = np.zeros(len(llc_coord), dtype='bool')
    good = sep2d < hp.pixel_resolution
    
    # Return the cut table
    for ii in np.where(good)[0]:
        flag[idx[ii]] = True
    return df[flag]


def preproc_image(item, pdict):
    """
    Simple wrapper for preproc_field()

    Parameters
    ----------
    item : tuple
        field, idx
    pdict : dict
        Preprocessing dict

    Returns
    -------
    pp_field, idx, meta : np.ndarray, int, dict

    """
    # Unpack
    field, idx = item

    # Run
    pp_field, meta = pp_utils.preproc_field(field, None, **pdict)

    # Failed?
    if pp_field is None:
        return None

    # Return
    return pp_field.astype(np.float32), idx, meta



def extract_preproc_for_analysis(resol=0.5, preproc_root='llc_std', 
                                 field_size=(64,64), n_cores=10,
                                 valid_fraction=1., 
                                 outfile='LLC_uniform_preproc.h5'):
    # Preprocess options
    pdict = pp_io.load_options(preproc_root)

    # Setup for parallel
    map_fn = partial(preproc_image,
                     pdict=pdict)

    # Coordinate table
    coord_tbl = setup_coords(resol)

    # Loop on files
    load_path = f'/home/xavier/Projects/Oceanography/data/LLC/ThetaUVSalt'
    model_files = glob.glob(os.path.join(load_path, 'LLC4320*'))

    metadata = pandas.DataFrame()
    
    idx = 0
    pp_fields, meta, img_idx = [], [], []
    for filename in model_files[0:9]:
        ds = xr.load_dataset(filename)
        sst = ds.Theta.values
        # Load up the cutouts
        print("Loading up the cutouts")
        fields = []
        for r, c in zip(coord_tbl.row, coord_tbl.column):
            fields.append(sst[r:r+field_size[0], c:c+field_size[1]])
        print("Cutouts loaded for {}".format(filename))

        # Multi-process time
        sub_idx = np.arange(idx, idx+len(fields)).tolist()
        idx += len(fields)
        # 
        items = [item for item in zip(fields,sub_idx)]

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, items,
                                             chunksize=chunksize), total=len(items)))

        # Deal with failures
        answers = [f for f in answers if f is not None]

        # Slurp
        pp_fields += [item[0] for item in answers]
        img_idx += [item[1] for item in answers]
        meta += [item[2] for item in answers]

        # Update the metadata
        tmp_tbl = coord_tbl.copy()
        tmp_tbl['filename'] = os.path.basename(filename)
        metadata = metadata.append(tmp_tbl, ignore_index=True)

        del answers, fields, items
        ds.close()

    # Recast
    pp_fields = np.stack(pp_fields)
    pp_fields = pp_fields[:, None, :, :]  # Shaped for training

    
    print("After pre-processing, there are {} images ready for analysis".format(pp_fields.shape[0]))
    
    # TODO -- Move the following to preproc

    # Modify metadata
    metadata = metadata.iloc[img_idx]
    # Mu
    metadata['mean_temperature'] = [imeta['mu'] for imeta in meta]
    clms = list(metadata.keys())
    clms += ['mean_temperature']
    # Others
    for key in ['Tmin', 'Tmax', 'T90', 'T10']:
        if key in meta[0].keys():
            metadata[key] = [imeta[key] for imeta in meta]
            clms += [key]

    # Train/validation
    n = int(valid_fraction * pp_fields.shape[0])
    idx = shuffle(np.arange(pp_fields.shape[0]))
    valid_idx, train_idx = idx[:n], idx[n:]

    # ###################
    # Write to disk

    with h5py.File(outfile, 'w') as f:
        # Validation
        f.create_dataset('valid', data=pp_fields[valid_idx].astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', data=metadata.iloc[valid_idx].to_numpy(dtype=str).astype('S'))
        dset.attrs['columns'] = clms
        # Train
        if valid_fraction < 1:
            f.create_dataset('train', data=pp_fields[train_idx].astype(np.float32))
            dset = f.create_dataset('train_metadata', data=metadata.iloc[train_idx].to_numpy(dtype=str).astype('S'))
            dset.attrs['columns'] = clms
    print("Wrote: {}".format(outfile))

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    if flg & (2**0):  
        extract_preproc_for_analysis()

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # outliers
    else:
        flg = sys.argv[1]

    main(flg)
