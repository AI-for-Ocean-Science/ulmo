""" Script to extract cutouts from LLC data """

import os
import numpy as np
import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from tqdm import tqdm

import xarray as xr

from ulmo import io as ulmo_io
from ulmo.preproc import utils as pp_utils
from ulmo.preproc import extract

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Ulmo LLC extraction')
    parser.add_argument('--clear_threshold', type=float, default=99.99,  # No clouds, but 100 breaks
                        help='Percent of field required to be clear')
    parser.add_argument('--field_size', type=int, default=64,
                        help='Pixel width/height of field')
    parser.add_argument('--temp_lower_bound', type=float, default=-2.,
                        help='Minimum temperature considered')
    parser.add_argument('--temp_upper_bound', type=float, default=33.,
                        help='Maximum temperature considered')
    parser.add_argument('--nrepeat', type=int, default=1,
                        help='Repeats for each good block')
    parser.add_argument('--nmin_patches', type=int, default=2,
                        help='Mininum number of random patches to consider from each file')
    #parser.add_argument('--nmax_patches', type=int, default=1000,
    #                    help='Maximum number of random patches to consider from each file')
    parser.add_argument('--ncores', type=int, help='Number of cores for processing')
    parser.add_argument('--nsub_files', type=int, default=100, 
        help='Number of files to process at a time')
    parser.add_argument("--debug", default=False, action="store_true", help="Debug?")
    args = parser.parse_args()

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def extract_file(ifile, load_path, latitude=None, longitude=None,
                 field_size=(64,64),
                 nadir_offset=None,
                 CC_max=0.05,
                 temp_bounds=(-3, 34),
                 nrepeat=1,
                 inpaint=False, debug=False):

    filename = os.path.join(load_path, ifile)

    # Load the image
    #sst, qual = ulmo_io.load_nc(filename, verbose=False)
    ds = xr.load_dataset(filename)
    sst = ds.Theta.values

    # Generate the masks
    masks = pp_utils.build_mask(sst, None, temp_bounds=temp_bounds)

    # Restrict to near nadir
    if nadir_offset is not None:
        nadir_pix = sst.shape[1] // 2
        lb = nadir_pix - nadir_offset
        ub = nadir_pix + nadir_offset
        sst = sst[:, lb:ub]
        masks = masks[:, lb:ub].astype(np.uint8)
    else:
        lb = 0

    # Random clear rows, cols
    rows, cols, clear_fracs = extract.clear_grid(masks, field_size[0], 'center',
                                                 CC_max=CC_max, nsgrid_draw=nrepeat)
    if rows is None:
        raise ValueError("No rows.  Something went wrong")

    # Extract
    fields, field_masks = [], []
    metadata = []
    for r, c, clear_frac in zip(rows, cols, clear_fracs):
        # Cut
        field = sst[r:r+field_size[0], c:c+field_size[1]]
        mask = masks[r:r+field_size[0], c:c+field_size[1]]
        # Append SST and mask
        fields.append(field.astype(np.float32))
        field_masks.append(mask)
        # meta
        row, col = r, c + lb
        lat = latitude[row + field_size[0] // 2, col + field_size[1] // 2]
        lon = longitude[row + field_size[0] // 2, col + field_size[1] // 2]
        metadata.append([ifile, str(row), str(col), str(lat), str(lon), str(clear_frac)])

    del sst, masks

    return np.stack(fields), np.stack(field_masks), np.stack(metadata)

def main(pargs):
    """ Run
    """
    # Filenames
    load_path = f'/home/xavier/Projects/Oceanography/data/LLC/ThetaUVSalt'
    save_path = os.path.join(os.getenv('SST_OOD'), f'LLC', f'Extractions',
        f'LLC_{int(pargs.clear_threshold)}clear_{pargs.field_size}x{pargs.field_size}.h5')

    # Load up latitude, longitude
    coord_file = f'/home/xavier/Projects/Oceanography/data/LLC/LLC_coords.nc'
    coord_ds = xr.load_dataset(coord_file)
    longitude = coord_ds.lon.values
    latitude = coord_ds.lat.values
    print('Coordinates loaded..')

    # Setup for preproc
    map_fn = partial(extract_file,
                     longitude=longitude,
                     latitude=latitude,
                     load_path=load_path,
                     field_size=(pargs.field_size, pargs.field_size),
                     CC_max=1.- pargs.clear_threshold/100.,
                     temp_bounds=(pargs.temp_lower_bound, pargs.temp_upper_bound),
                     nrepeat=pargs.nrepeat,
                     debug=pargs.debug)


    if pargs.debug:
        files = [f for f in os.listdir(load_path) if f.endswith('.nc')]
        files = files[0:1]
        answers = []
        for kk, ifile in enumerate(files):
            answers.append(extract_file(ifile,
                     longitude=longitude,
                     latitude=latitude,
                     load_path=load_path,
                     field_size=(pargs.field_size, pargs.field_size),
                     CC_max=1.- pargs.clear_threshold/100.,
                     temp_bounds=(pargs.temp_lower_bound, pargs.temp_upper_bound),
                     nrepeat=pargs.nrepeat))
            print("kk: {}".format(kk))
        # Unpack and save
        answers = [f for f in answers if f is not None]
        fields = np.concatenate([item[0] for item in answers])
        masks = np.concatenate([item[1] for item in answers])
        metadata = np.concatenate([item[2] for item in answers])

        # Write
        columns = ['filename', 'row', 'column', 'latitude', 'longitude', 'clear_fraction']

        save_path = os.path.join(os.getenv('SST_OOD'), f'LLC', f'Extractions', 
                                 f'debug_one_for_coords.h5')
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('fields', data=fields)
            f.create_dataset('masks', data=masks.astype(np.uint8))
            dset = f.create_dataset('metadata', data=metadata.astype('S'))
            dset.attrs['columns'] = columns
    embed(header='165 of exLLC')


    if pargs.ncores is None:
        n_cores = multiprocessing.cpu_count()
    else:
        n_cores= pargs.ncores
    print("Using: {} cores".format(n_cores))

    # Limit number of files to 10000
    files = [f for f in os.listdir(load_path) if f.endswith('.nc')]
    if pargs.wolverine:
        files = [f for f in os.listdir(load_path) if f.endswith('.nc')] * 50
    elif pargs.debug:
        #files = files[6000:8000]
        files = files[0:1000]

    nloop = len(files) // pargs.nsub_files + ((len(files) % pargs.nsub_files) > 0)
    print('Processing {} files in {} loops of {}'.format(len(files), nloop, pargs.nsub_files))
    embed(header='161 of exLLC')

    fields, masks, metadata = None, None, None
    for kk in range(nloop):
        i0 = kk*pargs.nsub_files
        i1 = min((kk+1)*pargs.nsub_files, len(files))
        print('Files: {}:{} of {}'.format(i0, i1, len(files)))
        sub_files = files[i0:i1]

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(sub_files) // n_cores if len(sub_files) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, sub_files,
                                             chunksize=chunksize), total=len(sub_files)))

        # Trim None's
        answers = [f for f in answers if f is not None]
        if fields is None:
            fields = np.concatenate([item[0] for item in answers])
            masks = np.concatenate([item[1] for item in answers])
            metadata = np.concatenate([item[2] for item in answers])
        else:
            fields = np.concatenate([fields]+[item[0] for item in answers], axis=0)
            masks = np.concatenate([masks]+[item[1] for item in answers], axis=0)
            metadata = np.concatenate([metadata]+[item[2] for item in answers], axis=0)
        del answers

    # Write
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 'clear_fraction']

    with h5py.File(save_path, 'w') as f:
        #f.create_dataset('fields', data=fields.astype(np.float32))
        f.create_dataset('fields', data=fields)
        f.create_dataset('masks', data=masks.astype(np.uint8))
        dset = f.create_dataset('metadata', data=metadata.astype('S'))
        dset.attrs['columns'] = columns
