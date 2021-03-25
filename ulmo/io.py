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

import boto3


def list_of_bucket_files(bucket_name, prefix='/', delimiter='/'):
    prefix = prefix[1:] if prefix.startswith(delimiter) else prefix
    bucket = s3.Bucket(bucket_name)
    return list(_.key for _ in bucket.objects.filter(Prefix=prefix))                                

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

def load_main_table(tbl_file:str, verbose=True):
    """Load the table of cutouts 

    Args:
        tbl_file (str): Path to table of cutouts. Local or s3
        verbose (bool, optional): [description]. Defaults to True.

    Raises:
        IOError: [description]

    Returns:
        pandas.DataFrame: table of cutouts
    """
    _, file_extension = os.path.splitext(tbl_file)
    # Allow for various formats
    if file_extension == '.csv':
        main_table = pandas.read_csv(tbl_file, index_col=0)
        # Set time
        if 'datetime' in main_table.keys():
            main_table.datetime = pandas.to_datetime(main_table.datetime)
    elif file_extension == '.feather':
        # Allow for s3
        with open(tbl_file, 'rb') as f:
            main_table = pandas.read_feather(f)
    else:
        raise IOError("Bad table extension: ")
    # Report
    if verbose:
        print("Read main table: {}".format(tbl_file))
    return main_table


def write_main_table(main_table:pandas.DataFrame, outfile:str, to_s3=True):
    """Write Main table for ULMO analysis
    Format is determined from the outfile extension.
        Options are ".csv", ".feather", ".parquet"

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
        bytes_ = BytesIO()
        main_table.to_feather(path=bytes_)
        if to_s3:
            write_bytes_to_s3(bytes_, outfile)
        else:
            write_bytes_to_local(bytes_, outfile)
    elif file_extension == '.parquet':
        bytes_ = BytesIO()
        main_table.to_parquet(path=bytes_)
        if to_s3:
            write_bytes_to_s3(bytes_, outfile)
        else:
            write_bytes_to_local(bytes_, outfile)
    else:
        raise IOError("Not ready for this")
    print("Wrote Analysis Table: {}".format(outfile))

def upload_file_to_s3(local_file:str, s3_uri:str):
    """Upload a single file to s3 storage

    Args:
        local_file (str): path to local file
        s3_uri (str): URL for s3 file 
    """
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
    parsed_s3 = urlparse(s3_uri)
    s3.meta.client.upload_file(local_file,
                             parsed_s3.netloc, 
                             parsed_s3.path[1:])
    
def write_bytes_to_local(bytes_:BytesIO, outfile:str):
    """Write a binary object to disk

    Args:
        bytes_ (BytesIO): contains the binary object
        outfile (str): [description]
    """
    bytes_.seek(0)
    with open(outfile, 'wb') as f:
        f.write(bytes_.getvalue())


def write_bytes_to_s3(bytes_:BytesIO, s3_uri:str):
    """Write bytes to s3 

    Args:
        bytes_ (BytesIO): contains the binary object
        s3_uri (str): Path to s3 bucket including filename
    """
    bytes_.seek(0)
    # Do it
    parsed_s3 = urlparse(s3_uri)
    s3.meta.client.upload_fileobj(Fileobj=bytes_, 
                             Bucket=parsed_s3.netloc, 
                             Key=parsed_s3.path[1:])