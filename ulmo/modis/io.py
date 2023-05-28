""" I/O for MODIS """

import numpy as np
import xarray

from ulmo.preproc import utils as pp_utils

def load_granule(filename:str,
                 field:str='SST',
                 qual_thresh=2,
                 temp_bounds = (-2, 33)):
    """ Load a MODIS granule

    Args:
        filename (str): MODIS filename
        field (str, optional): Field to use. Defaults to 'SST'. 
        qual_thresh (int, optional): Quality threshold. Defaults to 2.
        temp_bounds (tuple, optional): 
            Temperature bounds. Defaults to (-2, 33)

    Raises:
        IOError: _description_

    Returns:
        tuple: sst, latitude, longitude, masks
    """

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
