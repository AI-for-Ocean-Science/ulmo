""" Basic I/O methods"""

import numpy as np
import xarray as xr

def load_nc(filename, field='SST', verbose=True):
    """
    Load a MODIS or equivalent .nc file

    Parameters
    ----------
    filename : str
    field : str, optional
    verbose : bool, optional

    Returns
    -------
    field, qual, latitude, longitude : np.ndarray, np.ndarray, np.ndarray np.ndarray
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

    # Translate user field to MODIS
    mfields = dict(SST='sst', aph_443='aph_443_giop')

    # Flags
    mflags = dict(SST='qual_sst', aph_443='l2_flags')

    # Go for it
    try:
        # Fails if data is corrupt
        dfield = np.array(geo[mfields[field]])
        qual = np.array(geo[mflags[field]])
        latitude = np.array(nav['latitude'])
        longitude = np.array(nav['longitude'])
    except:
        if verbose:
            print("Data is corrupt!")
        return None, None, None, None

    geo.close()
    nav.close()

    # Return
    return dfield, qual, latitude, longitude

