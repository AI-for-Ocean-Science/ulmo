""" VIIRS specific utiliies """

def viirs_uid(df:pandas.DataFrame):
    """ Generate a unique identifier for VIIRS

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