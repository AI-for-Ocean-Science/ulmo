import os
import h5py
import argparse
import multiprocessing
import numpy as np
import xarray as xr
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from sklearn.utils import shuffle

from ulmo.preproc import utils as pp_utils
from ulmo import io


def extract_patches_2d(image, patch_size, n_patches):
    rows = np.random.choice(image.shape[0]-patch_size[0], size=n_patches)
    cols = np.random.choice(image.shape[1]-patch_size[1], size=n_patches)
    
    patches, locations = [], []
    for r, c in zip(rows, cols):
        patches.append(image[r:r+patch_size[0], c:c+patch_size[1]])
        locations.append([r, c])
    
    return np.stack(patches), np.array(locations)


def extract_fields(f, load_path, field_size=(128, 128), clear_thresh=0.95, qual_thresh=2, 
                   nadir_offset=480, temp_bounds=(-2, 33), n_patches=3000):

    # I/O
    filename = os.path.join(load_path, f)
    sst, qual, latitude, longitude = io.load_nc(filename, verbose=False)

    # Mask
    masks = pp_utils.build_mask(sst, qual, qual_thresh=qual_thresh, temp_bounds=temp_bounds)

    # Restrict to near nadir
    nadir_pix = sst.shape[1] // 2
    lb = nadir_pix - nadir_offset
    ub = nadir_pix + nadir_offset
    sst = sst[:, lb:ub]
    masks = masks[:, lb:ub]
    
    stacked = np.stack([sst, masks], axis=-1)
    fields, locs = extract_patches_2d(stacked, field_size, n_patches)
    fields, masks = fields[..., 0], fields[..., 1]

    clear_fields, metadata = [], []
    for field, mask, loc in zip(fields, masks, locs):
        clear_frac = 1. - mask.sum() / field.size
        if clear_frac >= clear_thresh:
            # Standard
            field, mu = pp_utils.preproc_field(field, mask)
            if field is None:  # Bad inpainting or the like
                continue
            # Get latitude/longitude at center of field
            row, col = loc[0], loc[1]+lb
            lat = latitude[row+field_size[0]//2, col+field_size[1]//2]
            lon = longitude[row+field_size[0]//2, col+field_size[1]//2]
            # Save
            clear_fields.append(field)
            metadata.append([f, str(row), str(col), str(lat), str(lon), str(mu), str(clear_frac)])
    
    if len(clear_fields) == 0:
        return
    else:
        print("f: {}, nclear = {}, frac={}".format(f, len(clear_fields)))
        return np.stack(clear_fields), np.stack(metadata)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ulmo Preprocessing')
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
    parser.add_argument('--n_patches', type=int, default=3000,
                        help='Number of random patches to consider from each file')  
    parser.add_argument('--valid_fraction', type=float, default=0.05,
                        help='Fraction of fields to holdout for validation')
    parser.add_argument("--debug", default=False, action="store_true", help="Debug?")
    args = parser.parse_args()
    
    load_path = f'/Volumes/Aqua-1/MODIS/night/night/{args.year}'
    save_path = (f'/Volumes/Aqua-1/MODIS/uri-ai-sst/xavier/MODIS_{args.year}'
        f'_{args.clear_threshold}clear_{args.field_size}x{args.field_size}.h5')

    map_fn = partial(extract_fields, load_path=load_path, field_size=(args.field_size, args.field_size),
                     clear_thresh=args.clear_threshold/100., qual_thresh=args.quality_threshold,
                     nadir_offset=args.nadir_offset, temp_bounds=(args.temp_lower_bound, args.temp_upper_bound),
                     n_patches=args.n_patches)

    n_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        files = [f for f in os.listdir(load_path) if f.endswith('.nc')]
        if args.debug:
            files = files[0:100]
        chunksize = len(files)//n_cores if len(files)//n_cores > 0 else 1
        fields = list(tqdm(executor.map(map_fn, files, chunksize=chunksize), total=len(files)))
        fields, metadata = np.array([f for f in fields if f is not None]).T
        fields, metadata = np.concatenate(fields), np.concatenate(metadata)

    if args.debug:
        from IPython import embed; embed(header='114 of preproc')
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