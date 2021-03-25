""" Script to pre-process images in an HDF5 file """

import numpy as np
import os
import json

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import h5py
from tqdm import tqdm

import pandas

from ulmo import io as ulmo_io
from ulmo.preproc import utils as pp_utils
from ulmo.preproc import io as pp_io

from sklearn.utils import shuffle

from IPython import embed


def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Preproc images in an H5 file.')
    parser.add_argument("infile", type=str, help="H5 file for pre-processing")
    parser.add_argument("valid_fraction", type=float, 
                        help="Validation fraction.  Can be 1")
    parser.add_argument("preproc_root", type=str,
                        help="Root name of JSON file containing the steps to be applied (standard, gradient)")
    parser.add_argument("outfile", type=str, help="H5 outfile name")
    parser.add_argument('--ncores', type=int, help='Number of cores for processing')
    parser.add_argument('--nsub_fields', type=int, default=10000,
                        help='Number of fields to parallel process at a time')
    parser.add_argument("--debug", default=False, action="store_true", help="Debug?")
    parser.add_argument("--kludge", default=False, action="store_true", help="X Kludge")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def preproc_image(item, pdict):
    """
    Simple wrapper for preproc_field()

    Parameters
    ----------
    item : tuple
        field, mask, idx
    pdict : dict
        Preprocessing dict

    Returns
    -------
    pp_field, idx, meta : np.ndarray, int, dict

    """
    # Parse
    field, mask, idx = item

    # Run
    pp_field, meta = pp_utils.preproc_field(field, mask, **pdict)

    # Failed?
    if pp_field is None:
        return None

    # Return
    return pp_field, idx, meta


def main(pargs):
    """ Run
    """
    import warnings

    if pargs.infile[-3:] != '.h5':
        print("infile must have .h5 extension")
        return

    # Open h5 file
    f = h5py.File(pargs.infile, mode='r')

    # Metadata
    meta = f['metadata']
    # Hack for bug
    clms = meta.attrs['columns'].tolist()
    if 'mean_temperature' in clms:
        clms.remove('mean_temperature')
    metadata = pandas.DataFrame(meta[:].astype(np.unicode_),
                                columns=clms)

    # Define train/validation here using MODIS

    # Pre-processing dict
    pdict = pp_io.load_options(pargs.preproc_root)

    # Setup for parallel
    map_fn = partial(preproc_image,
                     pdict=pdict)

    if pargs.ncores is None:
        n_cores = multiprocessing.cpu_count()
    else:
        n_cores= pargs.ncores

    # Prepare to loop
    nimages = f['fields'].shape[0]
    if pargs.debug:
        nimages = 1024
    nloop = nimages // pargs.nsub_fields + ((nimages % pargs.nsub_fields) > 0)

    print("There are {} images to process in the input file".format(nimages))
    f.close()


    # Process them all, then deal with train/validation
    pp_fields, meta, img_idx = [], [], []
    for kk in range(nloop):
        f = h5py.File(pargs.infile, mode='r')

        # Load the images into memory
        i0 = kk*pargs.nsub_fields
        i1 = min((kk+1)*pargs.nsub_fields, nimages)
        print('Fields: {}:{} of {}'.format(i0, i1, nimages))
        fields = f['fields'][i0:i1]
        shape =fields.shape
        masks = f['masks'][i0:i1].astype(np.uint8)
        sub_idx = np.arange(i0, i1).tolist()


        # Convert to lists
        print('Making lists')
        fields = np.vsplit(fields, shape[0])
        fields = [field.reshape(shape[1:]) for field in fields]

        if pargs.kludge:
            masks = [None]*len(fields)
        else:
            masks = np.vsplit(masks, shape[0])
            masks = [mask.reshape(shape[1:]) for mask in masks]

        items = [item for item in zip(fields,masks,sub_idx)]

        print('Process time')
        # Do it
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

        del answers, fields, masks, items
        f.close()

    # Recast
    pp_fields = np.stack(pp_fields)
    pp_fields = pp_fields[:, None, :, :]  # Shaped for training

    print("After pre-processing, there are {} images ready for analysis".format(pp_fields.shape[0]))

    # Modify metadata
    metadata = metadata.iloc[img_idx]
    if 'only_inpaint' in pdict.keys() and not pdict['only_inpaint']:
        # Mu
        metadata['mean_temperature'] = [imeta['mu'] for imeta in meta]
        clms += ['mean_temperature']
        # Others
        for key in ['Tmin', 'Tmax', 'T90', 'T10']:
            if key in meta[0].keys():
                metadata[key] = [imeta[key] for imeta in meta]
                clms += [key]

    if pargs.debug:
        embed(header='160 of preproc')

    # Train/validation
    n = int(pargs.valid_fraction * pp_fields.shape[0])
    idx = shuffle(np.arange(pp_fields.shape[0]))
    valid_idx, train_idx = idx[:n], idx[n:]

    # ###################
    # Write to disk

    with h5py.File(pargs.outfile, 'w') as f:
        # Validation
        f.create_dataset('valid', data=pp_fields[valid_idx].astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', data=metadata.iloc[valid_idx].to_numpy(dtype=str).astype('S'))
        dset.attrs['columns'] = clms
        # Train
        if pargs.valid_fraction < 1:
            f.create_dataset('train', data=pp_fields[train_idx].astype(np.float32))
            dset = f.create_dataset('train_metadata', data=metadata.iloc[train_idx].to_numpy(dtype=str).astype('S'))
            dset.attrs['columns'] = clms

    print("Wrote: {}".format(pargs.outfile))

