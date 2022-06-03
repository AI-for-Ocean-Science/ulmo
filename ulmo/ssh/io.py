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
    ssh, latitude, longitude : np.ndarray, np.ndarray, np.ndarray np.ndarray
        Height map
        Latitutides
        Longitudes
        or None's if the data is corrupt!
    """
    if filename[0:5] == 's3://':
        #inp = ulmo_io.load_to_bytes(filename)
        with ulmo_io.open(filename, 'rb') as f:
            ds = xarray.open_dataset(filename_or_obj=f,
                engine='netcdf4',
                mask_and_scale=True)
    else:
        inp = filename
        ds = xarray.open_dataset(filename_or_obj=inp,
            engine='netcdf4',
            mask_and_scale=True)
    #print(ds)
    try:
        # Fails if data is corrupt
        ssh = ds.SLA.data[0,...]
        latitude = ds.Latitude.data[:]
        longitude = ds.Longitude.data[:]
        
    except:
        if verbose:
            print("Data is corrupt!")
        return None, None, None, None

    ds.close()

    # Return
    return ssh, latitude, longitude

#fn = "https://opendap.jpl.nasa.gov/opendap/SeaSurfaceTopography/merged_alt/L4/cdr_grid/ssh_grids_v1812_1992100212.nc"
#a,b,c = (load_nc(fn))

#print(c)
