import os
import datetime

import numpy as np

import pandas

from IPython import embed

def times_from_filenames(filenames:list, ioff=10, toff=1):
    """ Generate list of datetimes from
    list of filenames
    
    Works for MODIS and VIIRS

    Args:
        filenames (list): List of filenames.
            Need to be the base (no path)
        ioff (int, optional): Offset in filename for timestamp. Defaults to 10.
        toff (int, optional): Offset for HMS, e.g. presence of a T
            Defaults to 1 for MODIS

    Returns:
        list:  List of datetimes
    """
    # Dates
    dtimes = [datetime.datetime(int(ifile[1+ioff:5+ioff]), # Year
                                int(ifile[5+ioff:7+ioff]), # Month
                                int(ifile[7 + ioff:9+ioff]), # Day
                                int(ifile[9+ioff+toff:11+ioff+toff]), # Hour
                                int(ifile[11+ioff+toff:13+ioff+toff])) # Minut
                for ifile in filenames]
    # Return
    return dtimes

def modis_uid(df:pandas.DataFrame):
    """ Generate a unique identifier for MODIS

    Args:
        df (pandas.DataFrame): main table

    Returns:
        numpy.ndarray: int64 array of unique identifiers
    """
    # Date?
    if 'date' not in df.keys():
        # Dates
        ioff = 10
        dtimes = [datetime.datetime(int(ifile[1+ioff:5+ioff]),
                                int(ifile[5+ioff:7+ioff]),
                                int(ifile[7 + ioff:9+ioff]),
                                int(ifile[10+ioff:12+ioff]),
                                int(ifile[12+ioff:14+ioff]))
                for ifile in df['filename'].values]
        df['date'] = dtimes
        
    # Unique identifier
    tlong = df['date'].values.astype(np.int64) // 10000000000
    latkey = 'latitude' if 'latitude' in df.keys() else 'lat'
    lonkey = 'longitude' if 'longitude' in df.keys() else 'lon'
    lats = np.round((df[latkey].values.astype(float) + 90)*10000).astype(int)
    lons = np.round((df[lonkey].values.astype(float) + 180)*100000).astype(int)
    uid = [np.int64('{:s}{:d}{:d}'.format(str(t)[:-5],lat,lon))
            for t,lat,lon in zip(tlong, lats, lons)]
    if len(uid) != len(np.unique(uid)):
        embed(header='67 of results')

    # Return
    return np.array(uid).astype(np.int64)