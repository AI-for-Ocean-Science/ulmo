""" Basic I/O methods"""

import numpy as np
import xarray as xr

def load_nc(filename, verbose=True):
    geo = xr.open_dataset(
        filename_or_obj=filename,
        group='geophysical_data',
        engine='h5netcdf',
        mask_and_scale=True)
    nav = xr.open_dataset(
        filename_or_obj=filename,
        group='navigation_data',
        engine='h5netcdf',
        mask_and_scale=True)

    try:
        # Fails if data is corrupt
        sst = np.array(geo['sst'])
        qual = np.array(geo['qual_sst'])
        latitude = np.array(nav['latitude'])
        longitude = np.array(nav['longitude'])
    except:
        if verbose:
            print("Data is corrupt!")
        return None, None, None, None

    # Return
    return sst, qual, latitude, longitude

