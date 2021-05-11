""" I/O routines for VIIRS data """

import xarray

from ulmo import io as ulmo_io

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
    if filename[0:5] == 's3://':
        #inp = ulmo_io.load_to_bytes(filename)
        with ulmo_io.open(filename, 'rb') as f:
            ds = xarray.open_dataset(filename_or_obj=f,
                engine='h5netcdf',
                mask_and_scale=True)
    else:
        inp = filename
        ds = xarray.open_dataset(filename_or_obj=inp,
            engine='h5netcdf',
            mask_and_scale=True)

    try:
        # Fails if data is corrupt
        sst = ds.sea_surface_temperature.data[0,...] - 273.15 # Celsius!
        qual = ds.quality_level.data[0,...].astype(int)
        #qual = ds.l2p_flags.data[0,...]
        latitude = ds.lat.data[:]
        longitude = ds.lon.data[:]
    except:
        if verbose:
            print("Data is corrupt!")
        return None, None, None, None

    ds.close()

    # Return
    return sst, qual, latitude, longitude
