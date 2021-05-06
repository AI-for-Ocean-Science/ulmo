""" I/O routines for VIIRS data """

import xarray
import numpy as np

def load_nc(filename, verbose=True):
    """
    Load a VIIRS .nc file

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
    ds = xarray.open_dataset(
        filename_or_obj=filename,
        engine='h5netcdf',
        mask_and_scale=True)

    try:
        # Fails if data is corrupt
        sst = ds.sea_surface_temperature.data[0,...]
        qual = ds.l2p_flags.data[0,...]
        latitude = ds.lat.data[:]
        longitude = ds.lon.data[:]
    except:
        if verbose:
            print("Data is corrupt!")
        return None, None, None, None

    ds.close()

    # Return
    return sst, qual, latitude, longitude
