""" Module for MODIS analysis on daytime SST"""
import os
import numpy as np
import subprocess 

import pandas
import h5py 

from ulmo import io as ulmo_io
from ulmo.preproc import io as pp_io 
from ulmo.modis import extract as modis_extract
from ulmo.modis import utils as modis_utils
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.utils import catalog as cat_utils

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

from IPython import embed

tbl_file = 's3://modis-l2/Tables/MODIS_L2_day_2011_std.parquet'


def modis_day_extract_2011(debug=False):

    n_cores = 10
    nsub_files = 20000
    # Pre-processing (and extraction) settings
    pdict = pp_io.load_options('standard')
    
    # 2011 
    load_path = '/data/Projects/Oceanography/data/MODIS/SST/day/2011/'
    files = [f for f in os.listdir(load_path) if f.endswith('.nc')]
    nloop = len(files) // nsub_files + ((len(files) % nsub_files) > 0)

    # Output
    save_path = ('MODIS_R2019_2011_day'
                 '_{}clear_{}x{}_inpaint.h5'.format(pdict['clear_threshold'],
                                                    pdict['field_size'],
                                                    pdict['field_size']))
    if debug:
        files = files[:100]

    # Setup for preproc
    map_fn = partial(modis_extract.extract_file,
                     load_path=load_path,
                     field_size=(pdict['field_size'], pdict['field_size']),
                     CC_max=1.-pdict['clear_threshold'] / 100.,
                     qual_thresh=pdict['quality_thresh'],
                     nadir_offset=pdict['nadir_offset'],
                     temp_bounds=tuple(pdict['temp_bounds']),
                     nrepeat=pdict['nrepeat'],
                     inpaint=True)

    
    fields, masks, metadata = None, None, None
    for kk in range(nloop):
        i0 = kk*nsub_files
        i1 = min((kk+1)*nsub_files, len(files))
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
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 
               'clear_fraction']

    # Local
    with h5py.File(save_path, 'w') as f:
        #f.create_dataset('fields', data=fields.astype(np.float32))
        f.create_dataset('fields', data=fields)
        f.create_dataset('masks', data=masks.astype(np.uint8))
        dset = f.create_dataset('metadata', data=metadata.astype('S'))
        dset.attrs['columns'] = columns

    # Table time
    modis_table = pandas.DataFrame()
    modis_table['filename'] = [item[0] for item in metadata]
    modis_table['row'] = [int(item[1]) for item in metadata]
    modis_table['col'] = [int(item[2]) for item in metadata]
    modis_table['lat'] = [float(item[3]) for item in metadata]
    modis_table['lon'] = [float(item[4]) for item in metadata]
    modis_table['field_size'] = pdict['field_size']
    modis_table['datetime'] = modis_utils.times_from_filenames(modis_table.filename.values)
    modis_table['clear_fraction'] = 1 - pdict['clear_threshold']/100.

    # Vet
    assert cat_utils.vet_main_table(modis_table)

    # Final write
    ulmo_io.write_main_table(modis_table, tbl_file)
    
    # Push to s3
    print("Pushing to s3")
    print("Run this:  s3 put {} s3://modis-l2/Extractions/{}".format(
        save_path, save_path))
    process = subprocess.run(['s4cmd', '--force', '--endpoint-url',
        'https://s3.nautilus.optiputer.net', 'put', save_path, 
           's3://modis-l2/Extractions/{}'.format(save_path)])


def modis_evaluate(test=True, noise=False):

    if test:
        tbl_file = tbl_test_noise_file if noise else tbl_test_file
    else:
        raise IOError("Not ready for anything but testing..")
    
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Rename
    if 'LL' in llc_table.keys() and 'modis_LL' not in llc_table.keys():
        llc_table = llc_table.rename(
            columns=dict(LL='modis_LL'))

    # Evaluate
    llc_table = ulmo_evaluate.eval_from_main(llc_table)

    # Write 
    assert cat_utils.vet_main_table(llc_table, cut_prefix='modis_')
    ulmo_io.write_main_table(llc_table, tbl_file)

def u_add_velocities():
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)
    
    # Velocities
    extract.velocity_stats(llc_table)

    # Write 
    ulmo_io.write_main_table(llc_table, tbl_file)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

        # MMT/MMIRS
    if flg & (2**0):
        modis_day_extract_2011(debug=False)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup coords
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate
        #flg += 2 ** 3  # 8 -- Init test + noise
        #flg += 2 ** 4  # 16 -- Extract + noise
        #flg += 2 ** 5  # 32 -- Evaluate + noise
    else:
        flg = sys.argv[1]

    main(flg)