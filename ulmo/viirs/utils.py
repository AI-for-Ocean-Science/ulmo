""" VIIRS specific utiliies """
import numpy as np

import datetime
import pandas

def viirs_uid(df:pandas.DataFrame):
    """ Generate a unique identifier for VIIRS

    Args:
        df (pandas.DataFrame): main table

    Returns:
        numpy.ndarray: int64 array of unique identifiers
    """
    # Unique identifier
    tlong = df['datetime'].values.astype(np.int64) // 10000000000
    latkey = 'latitude' if 'latitude' in df.keys() else 'lat'
    lonkey = 'longitude' if 'longitude' in df.keys() else 'lon'
    lats = np.round((df[latkey].values.astype(float) + 90)*10000).astype(int)
    lons = np.round((df[lonkey].values.astype(float) + 180)*100000).astype(int)
    uid = [np.int64('{:s}{:d}{:d}'.format(str(t)[:-5],lat,lon))
            for t,lat,lon in zip(tlong, lats, lons)]
    if len(uid) != len(np.unique(uid)):
        embed(header='67 of viirs/utils')

    # Return
    return np.array(uid).astype(np.int64)