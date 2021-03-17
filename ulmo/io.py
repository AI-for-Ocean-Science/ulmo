""" Basic I/O methods"""

import os
import numpy as np
import xarray as xr

import pandas
from urllib.parse import urlparse
from io import BytesIO

# DO NOT IMOPRT ANY ULMO!

# s3
import smart_open
import boto3
import functools

endpoint_url = (os.getenv('ENDPOINT_URL') 
                if os.getenv('ENDPOINT_URL') is not None else 
                    'http://rook-ceph-rgw-nautiluss3.rook')

s3 = boto3.resource('s3', endpoint_url=endpoint_url)
open = functools.partial(smart_open.open, 
                         transport_params={'resource_kwargs': 
                             {'endpoint_url': endpoint_url}})
                                

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

def load_main_table(tbl_file, verbose=True):
    _, file_extension = os.path.splitext(tbl_file)
    if file_extension == '.csv':
        main_table = pandas.read_csv(tbl_file, index_col=0)
    elif file_extension == '.feather':
        main_table = pandas.read_feather(tbl_file, index_col=0)
    else:
        raise IOError("Bad table extension: ")
    # Set time
    if 'datetime' in main_table.keys():
        llc_table.datetime = pandas.to_datetime(llc_table.datetime)
    return main_table


def write_main_table(main_table:pandas.DataFrame, outfile:str, to_s3=True):
    """Write Main table for ULMO analysis
    Format is determined from the outfile extension.
        Options are ".csv", ".feather"

    Args:
        main_table (pandas.DataFrame): Main table for ULMO analysis
        outfile (str): Output filename.  Its extension sets the format

    Raises:
        IOError: [description]
    """
    _, file_extension = os.path.splitext(outfile)
    if file_extension == '.csv':
        main_table.to_csv(outfile, date_format='%Y-%m-%d %H:%M:%S')
    elif file_extension == '.feather':
        if to_s3:
            write_pandas_to_s3_feather(main_table, s3_uri=outfile)
        else:
            main_table.to_feather(outfile) 
    else:
        raise IOError("Not ready for this")
    print("Wrote Analysis Table: {}".format(outfile))

    
def write_pandas_to_s3_feather(data:pandas.DataFrame, 
                               s3_uri:str, **kwargs):
    """Write pandas to s3 as a feather file

    Args:
        data (pandas.DataFrame): pandas table
        s3_uri (str): Path to s3 bucket including filename
        **kwargs: Passed to to_feather()
    """
    parsed_s3 = urlparse(s3_uri)
    bytes_ = BytesIO()
    data.to_feather(path=bytes_, **kwargs)
    bytes_.seek(0)
    # Do it
    s3.meta.client.upload_fileobj(Fileobj=bytes_, 
                             Bucket=parsed_s3.netloc, 
                             Key=parsed_s3.path[1:])

    print("Wrote: {}".format(s3_uri))