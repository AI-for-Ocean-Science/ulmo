""" Basic I/O methods"""

import os
import numpy as np
import xarray as xr

# DO NOT IMOPRT ANY ULMO!

# s3
import smart_open
import functools

open = functools.partial(smart_open.open, 
                         transport_params={'resource_kwargs': 
                             {'endpoint_url': 
                                 os.getenv('ENDPOINT_URL')}})

def load_nc(filename, verbose=True):
    """
    Load a MODIS or equivalent .nc file

    Parameters
    ----------
    filename : str
    verbose : bool, optional

    Returns
    -------
    sst, qual, latitude, longitude : np.ndarray, np.ndarray, np.ndarray np.ndarray
        Temperture map
        Quality
        Latitutides
        Longitudes
        or None's if the data is corrupt!

    """
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

    geo.close()
    nav.close()

    # Return
    return sst, qual, latitude, longitude

