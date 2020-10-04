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
from ulmo.preproc import io as pp_io



def cc_per_file(ifile, load_path, field_size=(128,128), nadir_offset=480,
                 qual_thresh=2, temp_bounds = (-2, 33), nrepeat=1,
                 inpaint=True, debug=False):

    filename = os.path.join(load_path, ifile)

    # Load the image
    try:
        sst, qual, latitude, longitude = ulmo_io.load_nc(filename, verbose=False)
    except:
        print("File {} is junk".format(filename))
        return
    if sst is None:
        return

    # Generate the masks
    masks = pp_utils.build_mask(sst, qual, qual_thresh=qual_thresh,
                                temp_bounds=temp_bounds)

    # Restrict to near nadir
    nadir_pix = sst.shape[1] // 2
    lb = nadir_pix - nadir_offset
    ub = nadir_pix + nadir_offset
    masks = masks[:, lb:ub].astype(np.uint8)

    # Loop on CC
    CC_values = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4,
                   0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    fracCC = np.zeros(len(CC_values))

    for ss, CC_max in enumerate(CC_values):

        # Random clear rows, cols
        fracCC[ss] = extract.clear_grid(masks, field_size[0], 'center',
                                        CC_max=CC_max, nsgrid_draw=nrepeat,
                                        return_fracCC=True)

    return ifile, fracCC


def main(n_cores, wolverine=False, debug=False):
    """ Run
    """
    # Filenames
    load_path = f'/Volumes/Aqua-1/MODIS_R2019/night/2010'
    save_path = f'/Volumes/Aqua-1/MODIS/uri-ai-sst/OOD/Analysis/cc_2010.h5'
    if wolverine:
        load_path = f'/home/xavier/Projects/Oceanography/AI/OOD/MODIS'
        save_path = './tst.h5'

    #
    pdict = pp_io.load_options('standard')

    # Setup for preproc
    map_fn = partial(cc_per_file,
                     load_path=load_path,
                     field_size=(pdict['field_size'], pdict['field_size']),
                     qual_thresh=pdict['quality_thresh'],
                     nadir_offset=pdict['nadir_offset'],
                     temp_bounds=tuple(pdict['temp_bounds']),
                     nrepeat=pdict['nrepeat'])


    # Limit number of files to 10000
    files = [f for f in os.listdir(load_path) if f.endswith('.nc')]
    nrand = 10000
    if wolverine:
        files = [f for f in os.listdir(load_path) if f.endswith('.nc')] * 50
        nrand = len(files)
        n_cores = 4
    elif debug:
        nrand = 1000

    # Grab them randomly
    idx = np.arange(len(files))
    np.random.shuffle(idx)

    files = np.array(files)[idx[0:nrand]].tolist()

    #print('Processing {} files in {} loops of {}'.format(len(files), nloop, pargs.nsub_files))

    fields, masks, metadata = None, None, None
    '''
    if wolverine:
        ifile, fracCC = cc_per_file(files[0],
                     load_path=load_path,
                     field_size=(pdict['field_size'], pdict['field_size']),
                     qual_thresh=pdict['quality_thresh'],
                     nadir_offset=pdict['nadir_offset'],
                     temp_bounds=tuple(pdict['temp_bounds']),
                     nrepeat=pdict['nrepeat'])
    '''
    print("Using: {} cores".format(n_cores))

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(files) // n_cores if len(files) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, files,
                                         chunksize=chunksize), total=len(files)))

    embed(header='119 of cc')
    # Trim None's
    answers = [f for f in answers if f is not None]
    # Slurp
    files = np.array([item[0] for item in answers])
    fracCC = np.stack([item[1] for item in answers])

    with h5py.File(save_path, 'w') as f:
        #f.create_dataset('fields', data=fields.astype(np.float32))
        f.create_dataset('files', data=files.astype('S'))
        f.create_dataset('fracCC', data=fracCC)

    print("Wrote: {}".format(save_path))

# Command line execution
if __name__ == '__main__':
    main(16, debug=True)
