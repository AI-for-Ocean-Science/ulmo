""" Analysis routines for MODIS """

import os
import glob

import datetime

import numpy as np
import pandas
from scipy.ndimage import uniform_filter
import h5py
import healpy as hp

from ulmo.modis import utils
from ulmo.modis import io as modis_io
from ulmo.preproc import extract

from IPython import embed

def build_main_from_old():
    """Turn the original MODIS table into a proper one
    """
    # On profx and the Google Drive but not s3

    old_file = os.path.join(os.getenv('SST_OOD'), 'MODIS_L2/Evaluations/R2010_results_std.feather')
    old_main = pandas.read_feather(old_file)

    # New one
    main = pandas.DataFrame()
    main['lat'] = old_main.latitude
    main['lon'] = old_main.longitude
    main['col'] = old_main.column
    main['datetime'] = old_main.date
    main['LL'] = old_main.log_likelihood

    # Same
    for key in ['row', 'clear_fraction', 'mean_temperature', 'Tmin',
                'Tmax', 'T90', 'T10', 'filename', 'UID']:
        main[key] = old_main[key]

    # New ones
    main['pp_file'] = ''
    main['pp_root'] = 'standard'
    main['field_size'] = 128
    main['pp_type'] = 0
    main['pp_idx'] = 0
    
    
    pp_dir = '/data/Projects/Oceanography/AI/OOD/SST/MODIS_L2/PreProc'
    pp_files = glob.glob(os.path.join(pp_dir, 
                         'MODIS_R2019_20*_95clear_128x128_preproc_std.h5'))
    pp_files.sort()                    
    for pp_file in pp_files:
        print("Working on: {}".format(pp_file))
        # Load up meta
        pp_hf2 = h5py.File(pp_file, 'r')
        meta = pp_hf2['valid_metadata']
        df = pandas.DataFrame(meta[:].astype(np.unicode_), 
                              columns=meta.attrs['columns'])
        pp_hf2.close()
        df['row'] = df.row.values.astype(int)
        df['column'] = df.column.values.astype(int)

        # Find location in main
        df0 = df.iloc[0]
        idx_main = np.where((main.row == df0.row) & (
            main.col == df0.column) & 
                            (main.filename == df0.filename))[0][0]
        # Check
        assert main.iloc[idx_main+len(df)-1].filename == df.iloc[-1].filename

        # Fill in
        idx = np.arange(len(df)) 
        main.loc[idx_main+idx,
                 'pp_file'] = 's3://modis-l2/PreProc/'+os.path.basename(pp_file)
        main.loc[idx_main+idx, 'pp_idx'] = idx


    # Add training
    pp_file = os.path.join(pp_dir, 
                           'MODIS_R2019_2010_95clear_128x128_preproc_std.h5')
    pp_hf2 = h5py.File(pp_file, 'r')
    meta = pp_hf2['train_metadata']
    df = pandas.DataFrame(meta[:].astype(np.unicode_), 
                            columns=meta.attrs['columns'])
    # Dates
    ioff = 10
    dtimes = [datetime.datetime(int(ifile[1+ioff:5+ioff]),
                                int(ifile[5+ioff:7+ioff]),
                                int(ifile[7 + ioff:9+ioff]),
                                int(ifile[10+ioff:12+ioff]),
                                int(ifile[12+ioff:14+ioff]))
                for ifile in df['filename'].values]

    # Unique identifier
    df['date'] = dtimes
    df['UID'] = utils.modis_uid(df)
    #df['UID'] = np.array(uid).astype(np.int64)


    train = pandas.DataFrame()
    train['row'] = df.row.values.astype(int)
    train['col'] = df.column.values.astype(int)
    train['filename'] = df.filename
    train['lat'] = df.latitude.values.astype(float)
    train['lon'] = df.longitude.values.astype(float)
    train['LL'] = np.nan
    train['datetime'] = dtimes
    train['UID'] = df.UID

    train['pp_file'] = 's3://modis-l2/PreProc/'+os.path.basename(pp_file)
    train['pp_root'] = 'standard'
    train['field_size'] = 128
    train['pp_type'] = 1
    train['pp_idx'] = np.arange(len(df))

    for key in ['clear_fraction', 'mean_temperature', 'Tmin',
                'Tmax', 'T90', 'T10']:
        train[key] = df[key].values.astype(float)

    # Append
    main = main.append(train, ignore_index=True)

    # Finish!
    outfile = os.path.join(os.getenv('SST_OOD'), 'MODIS_L2/Tables/MODIS_L2_std.feather')
    main.to_feather(outfile)

#build_main_from_old()


def cloud_cover_granule(filename:str, 
                        CC_values:list=None,  # Required but a parameter to allow for multiprocessing
                        nside:int=None, # Required but a parameter to allow for multiprocessing
                        field='SST',
                        field_size=(128,128),
                        nadir_offset=480,
                        qual_thresh=2,
                        temp_bounds = (-2, 33)):

    # Load
    sst, latitude, longitude, masks = modis_io.load_granule(
        filename, field=field, qual_thresh=qual_thresh,
        temp_bounds=temp_bounds)
    if sst is None:
        return

    # Restrict to near nadir
    nadir_pix = sst.shape[1] // 2
    lb = nadir_pix - nadir_offset
    ub = nadir_pix + nadir_offset
    masks = masks[:, lb:ub].astype(float)
    latitude = latitude[:, lb:ub]
    longitude = longitude[:, lb:ub]

    # Calculate the CC mask
    CC_mask, mask_edge = extract.clear_grid(
        masks, field_size[0], None, return_CC_mask=True)

    # Healpix
    idx_all = np.zeros_like(latitude, dtype=int) - 1  # init
    finite = np.isfinite(latitude) & np.isfinite(longitude)

    theta = (90 - latitude[finite]) * np.pi / 180.
    phi = longitude[finite] * np.pi / 180.
    idx_finite = hp.pixelfunc.ang2pix(nside, theta, phi)
    idx_all[finite] = idx_finite

    # Loop on em
    tot_pix = []
    hp_idx = []
    for CC_value in CC_values:
        # Clear
        clear = (CC_mask <= CC_value) & np.invert(mask_edge)  
        # Evaluate
        tot_pix.append(np.sum(clear))

        # Deal with bad values
        uni = np.unique(idx_all[clear])
        hp_idx.append(uni[uni>=0])

    # Return
    return tot_pix, hp_idx