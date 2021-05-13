""" Module for Ulmo analysis on VIIRS 2013"""
import os
import glob
import numpy as np
import subprocess 

import pandas
import h5py 

from sklearn.utils import shuffle

from ulmo import io as ulmo_io
from ulmo.preproc import io as pp_io 
from ulmo.preproc import utils as pp_utils
from ulmo.viirs import extract as viirs_extract
from ulmo.modis import utils as modis_utils
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.utils import catalog as cat_utils

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import subprocess
from tqdm import tqdm

from IPython import embed

tbl_file_2013 = 's3://viirs/Tables/VIIRS_2013_std.parquet'
s3_bucket = 's3://viirs'

def viirs_get_data_into_s3(debug=False, year=2013, day1=1):
    # Check
    assert os.getenv('PO_DAAC') is not None
    # Loop on days

    pushed_files = []
    nc_files = None

    # push to s3
    def push_to_s3(nc_files, sday, year):
        for nc_file in nc_files:
            s3_file = os.path.join(s3_bucket, 'data', str(year),
                                   sday, nc_file)
            ulmo_io.upload_file_to_s3(nc_file, s3_file)
            # Remove
            os.remove(nc_file)
    
    #for ss in range(365):
    ndays = 366
    for ss in range(day1-1, ndays):
        iday = ss + 1
        print("Working on day: {}".format(iday))
        sday = str(iday).zfill(3)
        # Popen
        pw = subprocess.Popen([
            'wget', '--no-check-certificate', '--user=profx', 
            '--password={}'.format(os.getenv('PO_DAAC')), 
            '-r', '-nc', '-np',  '-nH', '-nd', '-A', 
            '{}*.nc'.format(str(year)),
            #'*.nc', 
            'https://podaac-tools.jpl.nasa.gov/drive/files/allData/ghrsst/data/GDS2/L2P/VIIRS_NPP/OSPO/v2.61/{}/{}/'.format(
                year,sday)])
        if ss == 0:
            pass
        else:
            if len(nc_files) > 0:
                push_to_s3(nc_files, pvday, year)
        # Wait now
        pw.wait()
        # Files
        nc_files = glob.glob('{}*.nc'.format(year))
        nc_files.sort()
        pvday = sday
    # Last batch
    print("Pushing last batch")
    if len(nc_files) > 0:
        push_to_s3(nc_files, pvday, year)


def viirs_extract_2013(debug=False, n_cores=10):

    if debug:
        n_cores = 10
        tbl_file = 's3://viirs/Tables/VIIRS_2013_tst.parquet'
    else:
        tbl_file = tbl_file_2013
    nsub_files = 20000
    # Pre-processing (and extraction) settings
    pdict = pp_io.load_options('viirs_std')
    
    # 2013 
    print("Grabbing the file list")
    all_viirs_files = ulmo_io.list_of_bucket_files('viirs')
    files = []
    bucket = 's3://viirs/'
    for ifile in all_viirs_files:
        if 'data/2013' in ifile:
            files.append(bucket+ifile)
    nloop = len(files) // nsub_files + ((len(files) % nsub_files) > 0)

    # Output
    save_path = ('VIIRS_2013'
                 '_{}clear_{}x{}_inpaint.h5'.format(pdict['clear_threshold'],
                                                    pdict['field_size'],
                                                    pdict['field_size']))
    s3_filename = 's3://viirs/Extractions/{}'.format(save_path)

    if debug:
        # Grab 100 random
        files = shuffle(files, random_state=1234)
        files = files[:5]
        #files = files[:100]

    # Setup for preproc
    map_fn = partial(viirs_extract.extract_file,
                     field_size=(pdict['field_size'], pdict['field_size']),
                     CC_max=1.-pdict['clear_threshold'] / 100.,
                     nadir_offset=pdict['nadir_offset'],
                     temp_bounds=tuple(pdict['temp_bounds']),
                     nrepeat=pdict['nrepeat'],
                     sub_grid_step=pdict['sub_grid_step'],
                     inpaint=True)

    
    fields, inpainted_masks, metadata = None, None, None
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
            inpainted_masks = np.concatenate([item[1] for item in answers])
            metadata = np.concatenate([item[2] for item in answers])
        else:
            fields = np.concatenate([fields]+[item[0] for item in answers], axis=0)
            inpainted_masks = np.concatenate([inpainted_masks]+[item[1] for item in answers], axis=0)
            metadata = np.concatenate([metadata]+[item[2] for item in answers], axis=0)
        del answers

    # Write
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 
               'clear_fraction']

    # Local
    with h5py.File(save_path, 'w') as f:
        #f.create_dataset('fields', data=fields.astype(np.float32))
        f.create_dataset('fields', data=fields)
        f.create_dataset('inpainted_masks', data=inpainted_masks)#.astype(np.uint8))
        dset = f.create_dataset('metadata', data=metadata.astype('S'))
        dset.attrs['columns'] = columns

    # Table time
    viirs_table = pandas.DataFrame()
    viirs_table['filename'] = [item[0] for item in metadata]
    viirs_table['row'] = [int(item[1]) for item in metadata]
    viirs_table['col'] = [int(item[2]) for item in metadata]
    viirs_table['lat'] = [float(item[3]) for item in metadata]
    viirs_table['lon'] = [float(item[4]) for item in metadata]
    viirs_table['clear_fraction'] = [float(item[5]) for item in metadata]
    viirs_table['field_size'] = pdict['field_size']
    basefiles = [os.path.basename(ifile) for ifile in viirs_table.filename.values]
    viirs_table['datetime'] = modis_utils.times_from_filenames(basefiles, ioff=-1, toff=0)
    viirs_table['ex_filename'] = s3_filename

    # Vet
    assert cat_utils.vet_main_table(viirs_table)

    # Final write
    ulmo_io.write_main_table(viirs_table, tbl_file)
    
    # Push to s3
    print("Pushing to s3")
    ulmo_io.upload_file_to_s3(save_path, s3_filename)
    #print("Run this:  s3 put {} s3://modis-l2/Extractions/{}".format(
    #    save_path, save_path))
    #process = subprocess.run(['s4cmd', '--force', '--endpoint-url',
    #    'https://s3.nautilus.optiputer.net', 'put', save_path, 
    #    s3_filename])


def viirs_2013_preproc(test=False, debug=False):
    """Pre-process the files

    Args:
        test (bool, optional): [description]. Defaults to False.
    """
    if debug:
        tbl_file = 's3://viirs/Tables/VIIRS_2013_tst.parquet'
    else:
        tbl_file = tbl_file_2013
    viirs_tbl = ulmo_io.load_main_table(tbl_file)
    viirs_tbl = pp_utils.preproc_tbl(viirs_tbl, 1., 
                                     's3://viirs',
                                     preproc_root='viirs_std',
                                     inpainted_mask=True,
                                     use_mask=True)
    # Vet
    assert cat_utils.vet_main_table(viirs_tbl)

    # Final write
    ulmo_io.write_main_table(viirs_tbl, tbl_file)

def modis_day_evaluate(test=False):

    # Load
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Evaluate
    modis_tbl = ulmo_evaluate.eval_from_main(modis_tbl)

    # Write 
    assert cat_utils.vet_main_table(modis_tbl)
    ulmo_io.write_main_table(modis_tbl, tbl_file)


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # VIIRS download
    if flg & (2**0):
        viirs_get_data_into_s3(debug=False)

    # VIIRS extract
    if flg & (2**1):
        viirs_extract_2013(debug=True)

    # VIIRS preproc
    if flg & (2**2):
        viirs_2013_preproc(debug=True)

    # MODIS pre-proc
    #if flg & (2**2):
    #    modis_day_evaluate()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- VIIRS 2013 download
        #flg += 2 ** 1  # Extract
        flg += 2 ** 2  # Pre-proc
    else:
        flg = sys.argv[1]

    main(flg)