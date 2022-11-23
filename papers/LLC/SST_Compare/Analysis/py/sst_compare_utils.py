""" Analysis related methods """

import os
import numpy as np

import pandas

from ulmo import io as ulmo_io

s3_llc_match_table_file = 's3://llc/Tables/llc_viirs_match.parquet'
s3_llc_uniform_table_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
s3_viirs_table_file = 's3://viirs/Tables/VIIRS_all_98clear_std.parquet'
s3_modis_table_file = 's3://modis-l2/Tables/MODIS_SSL_96clear.parquet'

def load_table(dataset:str, local:bool=False, cut_lat_max:float=57.,
                cut_lat_min=-70., time_cut=None,
               cut_DT:tuple=None):
    """ Load the output table

    Args:
        dataset (str): 
            Dataset. Either [viirs, modis, llc_match or llc_uniform]
        local (bool, optional): 
            Load from local. Defaults to False.
        cut_lat (float, optional): 
            Cut on latitude. Defaults to 57..


    Returns:
        pandas.DataFrame: data
    """

    # Which flavor? 
    if dataset[0:3] == 'llc':
        if dataset == 'llc_match':
            s3_file = s3_llc_match_table_file
        elif dataset == 'llc_uniform':
            s3_file = s3_llc_uniform_table_file
        else:
            raise IOError("Bad llc dataset!")
        if local:
            tbl_file = os.path.join(os.getenv('SST_OOD'),
                'LLC', 'Tables', os.path.basename(s3_file))
        else:
            tbl_file = s3_file
    elif dataset == 'viirs':
        if local:
            tbl_file = os.path.join(os.getenv('SST_OOD'),
                'VIIRS', 'Tables', os.path.basename(s3_viirs_table_file))
        else:
            tbl_file = s3_viirs_table_file
    elif dataset == 'modis_all':
        tbl_file = s3_modis_table_file
    else:
        raise IOError("Bad Dataset")

    # Load
    tbl = ulmo_io.load_main_table(tbl_file)

    # Cut?
    if cut_lat_max is not None:
        tbl = tbl[tbl.lat < cut_lat_max].copy()

    if cut_lat_min is not None:
        tbl = tbl[tbl.lat > cut_lat_min].copy()

    if cut_DT is not None:
        tbl.DT = tbl.T90.values - tbl.T10.values
        tbl = tbl[(tbl.DT < cut_DT[1]) & (tbl.DT >= cut_DT[0])].copy()

    if time_cut == 'head':
        cutt = (tbl.datetime >= pandas.Timestamp(2012,2,1)) & (
            tbl.datetime< pandas.Timestamp(2016,1,31))
        tbl = tbl[cutt].copy()
    elif time_cut == 'tail':
        cutt = (tbl.datetime >= pandas.Timestamp(2017,1,1)) & (
            tbl.datetime < pandas.Timestamp(2020,12,31))
        tbl = tbl[cutt].copy()

    # Expunge Nan
    finite = np.isfinite(tbl.LL)
    tbl = tbl[finite]
    tbl.reset_index(drop=True, inplace=True)

    # Return
    return tbl