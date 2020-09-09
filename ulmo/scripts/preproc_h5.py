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

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Preproc images in an H5 file.')
    parser.add_argument("infile", type=str, help="H5 file for pre-processing")
    parser.add_argument("outfile", type=str, help="Output file.  Should have .h5 extension")
    parser.add_argument("--skip_inpaint", default=False, action="store_true",
                        help="Skip inpainting?")
    parser.add_argument('--ncores', type=int, help='Number of cores for processing')
    parser.add_argument('--nsub_fields', type=int, default=10000,
                        help='Number of fields to parallel process at a time')
    parser.add_argument("--debug", default=False, action="store_true", help="Debug?")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs

def preproc_image(item, pdict):

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


    # File check
    if pargs.outfile[-3:] != '.h5':
        print("outfile must have .h5 extension")
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

    # Pre-processing dict
    pdict = dict(inpaint=True, median=True, med_size=(3, 1),
                      downscale=True, dscale_size=(2, 2))

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
        nimages = 1000
    nloop = nimages // pargs.nsub_fields + ((nimages % pargs.nsub_fields) > 0)

    # Process them all, then deal with train/validation
    pp_fields, mu, img_idx = [], [], []
    for kk in range(nloop):
        # Load the images into memory
        i0 = kk*pargs.nsub_fields
        i1 = min((kk+1)*pargs.nsub_fields, nimages)
        print('Files: {}:{} of {}'.format(i0, i1, nimages))
        fields = f['fields'][i0:i1]
        masks = f['masks'][i0:i1].astype(bool)
        sub_idx = range(i0, i1)

        # Convert to lists
        items = []
        for i in range(fields.shape[0]):
            items.append([fields[i,...], masks[i,...], sub_idx[i]])
        del fields, masks

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

    # Close infile
    f.close()

    # Recast
    pp_fields = np.stack(pp_fields)

    # Modify metadata
    metadata = metadata.iloc[img_idx]
    metadata['mean_temperature'] = mu

    # ###################
    # Write to disk

    # First the pandas table
    metadata.to_hdf(pargs.outfile, 'metadata', mode='w')

    # Now the images
    newf = h5py.File(pargs.outfile, mode='a')
    # Add pre-processing steps
    for key in pdict:
        newf.attrs[key] = pdict[key]
    # Add images
    dset = newf.create_dataset("pp_fields", data=pp_fields)
    newf.close()

    print("Wrote: {}".format(pargs.outfile))
