""" I/O for MODIS """

import os
import numpy as np
import xarray

from ulmo.preproc import utils as pp_utils

def load_granule(filename:str,
                 field:str='SST',
                 qual_thresh=2,
                 temp_bounds = (-2, 33)):

    if filename[0:5] == 's3://':
        raise IOError("Not ready for s3 files yet. Multi-process is not working")
        #inp = ulmo_io.load_to_bytes(filename)
    else:
        geo = xarray.open_dataset(
                filename_or_obj=filename,
                group='geophysical_data',
                engine='h5netcdf',
                mask_and_scale=True)
        nav = xarray.open_dataset(
                filename_or_obj=filename,
                group='navigation_data',
                engine='h5netcdf',
                mask_and_scale=True)

    # Translate user field to MODIS
    mfields = dict(SST='sst', aph_443='aph_443_giop')

    # Flags
    mflags = dict(SST='qual_sst', aph_443='l2_flags')

    # Load the image
    try:
        sst = np.array(geo[mfields[field]])
        qual = np.array(geo[mflags[field]])
        latitude = np.array(nav['latitude'])
        longitude = np.array(nav['longitude'])
    except:
        print("File {} is junk".format(filename))
        return None, None, None, None
    if sst is None:
        return None, None, None, None

    # Generate the masks
    masks = pp_utils.build_mask(
        sst, qual, qual_thresh=qual_thresh,
        temp_bounds=temp_bounds)

    # Return
    return sst, latitude, longitude, masks


def grab_table_name(table:str=None, 
         local=False, cuts:str=None): 

    # Which file?
    if table is None:
        table = '96' 
    if table == 'std':  # Original; too many clouds
        basename = 'MODIS_L2_std.parquet'
    else:
        # Base 1
        if 'CF' in table:
            base1 = 'MODIS_SSL_cloud_free'
        elif '96_v4' in table:
            base1 = 'MODIS_SSL_v4'
        elif '96' in table:
            base1 = 'MODIS_SSL_96clear'
        # DT
        if 'DT' in table:
            if 'v4' in table:
                base1 = 'MODIS_SSL_96clear_v4'
            dtstr = table.split('_')[-1]
            base2 = '_'+dtstr
        elif 'v4_a' in table:
            base1 = 'MODIS_SSL_96clear_v4'
            dtstr = table.split('_')[-1]
            base2 = '_'+dtstr
        else:
            base2 = ''
        # 
        basename = base1+base2+'.parquet'

    if local:
        tbl_file = os.path.join(os.getenv('SST_OOD'), 'MODIS_L2', 'Tables', basename)
    else:
        tbl_file = 's3://modis-l2/Tables/'+basename

    return tbl_file
