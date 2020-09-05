""" Script to calculate LL for a field in a MODIS image"""

import os
import numpy as np
import h5py

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from tqdm import tqdm

from IPython import embed
from ulmo import io as ulmo_io
from ulmo.preproc import utils as pp_utils
from ulmo.preproc import extract


def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Ulmo MODIS extraction')
    parser.add_argument('--year', type=int, default=2010,
                        help='Data year to parse')
    parser.add_argument('--clear_threshold', type=int, default=95,
                        help='Percent of field required to be clear')
    parser.add_argument('--field_size', type=int, default=128,
                        help='Pixel width/height of field')
    parser.add_argument('--quality_threshold', type=int, default=2,
                        help='Maximum quality value considered')
    parser.add_argument('--nadir_offset', type=int, default=480,
                        help='Maximum pixel offset from satellite nadir')
    parser.add_argument('--temp_lower_bound', type=float, default=-2.,
                        help='Minimum temperature considered')
    parser.add_argument('--temp_upper_bound', type=float, default=33.,
                        help='Maximum temperature considered')
    parser.add_argument('--nmin_patches', type=int, default=10,
                        help='Mininum number of random patches to consider from each file')
    parser.add_argument('--nmax_patches', type=int, default=1000,
                        help='Maximum number of random patches to consider from each file')
    args = parser.parse_args()

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def extract_file(ifile, load_path, field_size=(128,128), nadir_offset=480,
                 CC_max=0.05, qual_thresh=2, temp_bounds = (-2, 33),
                 ndraw_mnx=(10,1000)):

    filename = os.path.join(load_path, ifile)

    # Load the image
    sst, qual, latitude, longitude = ulmo_io.load_nc(filename, verbose=False)

    # Generate the masks
    masks = pp_utils.build_mask(sst, qual, qual_thresh=qual_thresh,
                                temp_bounds=temp_bounds)

    # Restrict to near nadir
    nadir_pix = sst.shape[1] // 2
    lb = nadir_pix - nadir_offset
    ub = nadir_pix + nadir_offset
    sst = sst[:, lb:ub]
    masks = masks[:, lb:ub]

    # Random clear rows, cols
    rows, cols, clear_fracs = extract.random_clear(masks, field_size[0], CC_max=CC_max,
                                                   ndraw_mnx=ndraw_mnx)
    if rows is None:
        return

    # Extract
    fields, field_masks, locations = [], [], []
    metadata = []
    for r, c, clear_frac in zip(rows, cols, clear_fracs):
        # SST and mask
        fields.append(sst[r:r+field_size[0], c:c+field_size[1]])
        field_masks.append(masks[r:r+field_size[0], c:c+field_size[1]])
        # meta
        row, col = r, c + lb
        lat = latitude[row + field_size[0] // 2, col + field_size[1] // 2]
        lon = longitude[row + field_size[0] // 2, col + field_size[1] // 2]
        metadata.append([ifile, str(row), str(col), str(lat), str(lon), str(clear_frac)])

    #return np.stack(fields), np.stack(field_masks), np.stack(metadata)
    return np.stack(fields), np.stack(metadata)

def main(pargs):
    """ Run
    """
    #load_path = f'/Volumes/Aqua-1/MODIS/night/night/{pargs.year}'
    load_path = f'/home/xavier/Projects/Oceanography/AI/OOD'
    #save_path = (f'/Volumes/Aqua-1/MODIS/uri-ai-sst/dreiman/MODIS_{pargs.year}'
    save_path = (f'{load_path}_{pargs.year}'
                 f'_{pargs.clear_threshold}clear_{pargs.field_size}x{pargs.field_size}.h5')

    map_fn = partial(extract_file,
                     load_path=load_path,
                     field_size=(pargs.field_size, pargs.field_size),
                     CC_max=1.-pargs.clear_threshold / 100.,
                     qual_thresh=pargs.quality_threshold,
                     nadir_offset=pargs.nadir_offset,
                     temp_bounds=(pargs.temp_lower_bound, pargs.temp_upper_bound),
                     ndraw_mnx=(pargs.nmin_patches, pargs.nmax_patches))

    n_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        files = [f for f in os.listdir(load_path) if f.endswith('.nc')] *2
        chunksize = len(files) // n_cores if len(files) // n_cores > 0 else 1
        fields = list(tqdm(executor.map(map_fn, files, chunksize=chunksize), total=len(files)))

    embed(header='115 of extract_modis')
    fields, metadata = np.array([f for f in fields if f is not None]).T
    fields, masks, metadata = np.array([f for f in fields if f is not None]).T
    fields, masks, metadata = np.concatenate(fields), np.concatenate(masks), np.concatenate(metadata)

    embed(header='106 of extract_modis')

    fields = fields[:, None, :, :]
    n = int(args.valid_fraction * fields.shape[0])
    idx = shuffle(np.arange(fields.shape[0]))
    valid_idx, train_idx = idx[:n], idx[n:]
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 'mean_temperature', 'clear_fraction']

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('train', data=fields[train_idx].astype(np.float32))
        dset = f.create_dataset('train_metadata', data=metadata[train_idx].astype('S'))
        dset.attrs['columns'] = columns
        f.create_dataset('valid', data=fields[valid_idx].astype(np.float32))
        dset = f.create_dataset('valid_metadata', data=metadata[valid_idx].astype('S'))
        dset.attrs['columns'] = columns

