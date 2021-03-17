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
from ulmo.llc import extract
from ulmo.llc import io as llc_io

# Astronomy tools
import astropy_healpix
from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky

from sklearn.utils import shuffle

from IPython import embed


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



def extract_preproc_for_analysis(llc_table, preproc_root='llc_std', 
                                 field_size=(64,64), n_cores=10,
                                 valid_fraction=1., 
                                 outfile='LLC_uniform_preproc.h5'):
    # Preprocess options
    pdict = pp_io.load_options(preproc_root)

    # Setup for parallel
    map_fn = partial(preproc_image, pdict=pdict)

    # Setup for dates
    uni_date = np.unique(llc_table.datetime)
    if len(uni_date) > 10:
        raise IOError("You are likely to exceed the RAM.  Deal")

    # Init
    pp_fields, meta, img_idx = [], [], []

    # Loop
    for udate in uni_date:
        # Parse filename
        filename = llc_io.build_llc_datafile(udate)

        ds = xr.load_dataset(filename)
        sst = ds.Theta.values
        # Parse 
        gd_date = llc_table.datetime == udate
        sub_idx = np.where(gd_date)[0]
        coord_tbl = llc_table[gd_date]
        # Load up the cutouts
        fields = []
        for r, c in zip(coord_tbl.row, coord_tbl.col):
            fields.append(sst[r:r+field_size[0], c:c+field_size[1]])
        print("Cutouts loaded for {}".format(filename))

        # Multi-process time
        #sub_idx = np.arange(idx, idx+len(fields)).tolist()
        #idx += len(fields)
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
        #tmp_tbl = coord_tbl.copy()
        #tmp_tbl['filename'] = os.path.basename(filename)
        #metadata = metadata.append(tmp_tbl, ignore_index=True)

        del answers, fields, items
        ds.close()

    # Recast
    pp_fields = np.stack(pp_fields)
    pp_fields = pp_fields[:, None, :, :]  # Shaped for training

    print("After pre-processing, there are {} images ready for analysis".format(pp_fields.shape[0]))
    
    # Reorder llc_table (probably no change)
    llc_table = llc_table.iloc[img_idx]
    # Mu
    llc_table['mean_temperature'] = [imeta['mu'] for imeta in meta]
    clms = list(llc_table.keys())
    # Others
    for key in ['Tmin', 'Tmax', 'T90', 'T10']:
        if key in meta[0].keys():
            llc_table[key] = [imeta[key] for imeta in meta]
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
        dset = f.create_dataset('valid_metadata', data=llc_table.iloc[valid_idx].to_numpy(dtype=str).astype('S'))
        dset.attrs['columns'] = clms
        # Train
        if valid_fraction < 1:
            f.create_dataset('train', data=pp_fields[train_idx].astype(np.float32))
            dset = f.create_dataset('train_metadata', data=llc_table.iloc[train_idx].to_numpy(dtype=str).astype('S'))
            dset.attrs['columns'] = clms
    print("Wrote: {}".format(outfile))

    # Return
    return llc_table
