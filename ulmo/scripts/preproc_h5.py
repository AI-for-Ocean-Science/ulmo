""" Script to pre-process images in an HDF5 file """

import numpy as np
import os

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import h5py
from tqdm import tqdm

import pandas

from ulmo import io as ulmo_io
from ulmo.preproc import utils as pp_utils

from sklearn.utils import shuffle

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Preproc images in an H5 file.')
    parser.add_argument("infile", type=str, help="H5 file for pre-processing")
    parser.add_argument("valid_fraction", type=float, help="Validation fraction.  Can be 1")
    parser.add_argument("--inpaint", default=False, action="store_true", help="Inpaint?")
    parser.add_argument('--ncores', type=int, help='Number of cores for processing')
    parser.add_argument('--nsub_fields', type=int, default=10000,
                        help='Number of fields to parallel process at a time')
    parser.add_argument("--debug", default=False, action="store_true", help="Debug?")
    parser.add_argument("--only_inpaint", default=False, action="store_true", help="Only inpaint?")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs

def preproc_image(item, pdict):

    # Parse
    field, mask, idx = item

    # Run
    pp_field, mu = pp_utils.preproc_field(field, mask, **pdict)

    # Failed?
    if pp_field is None:
        return None

    # Return
    return pp_field, idx, mu

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
                                columns=clms) #meta.attrs['columns'])
    f.close()

    # Pre-processing dict
    pdict = dict(inpaint=pargs.inpaint,
                 median=True, med_size=(3, 1),
                 downscale=True, dscale_size=(2, 2),
                 only_inpaint=pargs.only_inpaint)

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

    # Process them all, then deal with train/validation
    pp_fields, mu, img_idx = [], [], []
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
        mu += [item[2] for item in answers]

        del answers, fields, masks, items
        f.close()

    # Recast
    pp_fields = np.stack(pp_fields)
    pp_fields = pp_fields[:, None, :, :]  # Shaped for training

    print("After pre-processing, there are {} images ready for analysis".format(pp_fields.shape[0]))

    # Modify metadata
    metadata = metadata.iloc[img_idx]
    if not pargs.only_inpaint:
        metadata['mean_temperature'] = mu

    # Train/validation
    n = int(pargs.valid_fraction * pp_fields.shape[0])
    idx = shuffle(np.arange(pp_fields.shape[0]))
    valid_idx, train_idx = idx[:n], idx[n:]


    # ###################
    # Write to disk
    suffix = '_preproc_{:.1f}valid.h5'.format(pargs.valid_fraction)
    outfile = pargs.infile.replace('.h5', suffix)

    with h5py.File(outfile, 'w') as f:
        # Validation
        f.create_dataset('valid', data=pp_fields[valid_idx].astype(np.float32))
        # Train
        if pargs.valid_fraction < 1:
            f.create_dataset('train', data=pp_fields[train_idx].astype(np.float32))
    # Metadata
    metadata.iloc[valid_idx].to_hdf(outfile, 'valid_metadata', mode='a')
    if pargs.valid_fraction < 1:
        metadata.iloc[train_idx].to_hdf(outfile, 'train_metadata', mode='a')

    print("Wrote: {}".format(outfile))

