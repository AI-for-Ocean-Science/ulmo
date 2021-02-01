""" Methods to slurp LLC data of interest"""
import xarray as xr


def write_xr(xr_d, outfile, strip_coord=True, encode=True):
    """
    Write an input xarray.DataArray of Theta to a netcdf file

    Parameters
    ----------
    xr_d : xarray.DataArray or xarray.DataSet
    outfile : str
    strip_coord  : bool, optional
        Strip off coordinates?
    encode : bool, optional
        Encode to int16?

    """

    # Strip coords?
    if strip_coord:
        drop_coord = []
        for key in xr_d.coords.keys():
            if key in ['i', 'j', 'i_g', 'j_g']:
                continue
            drop_coord.append(key)
        xr_d = xr_d.drop_vars(drop_coord)

    # Convert to Dataset
    if isinstance(xr_d, xr.DataArray):
        xr_ds = xr_d.to_dataset()
    elif isinstance(xr_d, xr.Dataset):
        xr_ds = xr_d
    else:
        raise IOError("Bad xr data type")

    # Encode?
    if encode:
        encoding = {}
        encoding['Theta'] = {'dtype': 'int16', 'scale_factor': 1e-3,
                             'add_offset': 10., 'zlib': True, 
                             '_FillValue': -32767,
                             'missing_value': -32768}
        encoding['U'] = {'dtype': 'int16', 'scale_factor': 1e-3,
                             'add_offset': 0., 'zlib': True, 
                             '_FillValue': -32767,
                             'missing_value': -32768}
        encoding['V'] = {'dtype': 'int16', 'scale_factor': 1e-3,
                             'add_offset': 0., 'zlib': True, 
                             '_FillValue': -32767,
                             'missing_value': -32768}
        encoding['W'] = {'dtype': 'int16', 'scale_factor': 1e-6,
                             'add_offset': 0., 'zlib': True, 
                             '_FillValue': -32767,
                             'missing_value': -32768}
        encoding['Salt'] = {'dtype': 'int16', 'scale_factor': 1e-3,
                             'add_offset': 30., 'zlib': True, 
                             '_FillValue': -32767,
                             'missing_value': -32768}
    else:
        encoding = None

    # Write
    xr_ds.to_netcdf(outfile, encoding=encoding)