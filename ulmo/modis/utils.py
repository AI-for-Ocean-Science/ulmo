import os
import datetime

def times_from_filenames(filenames:list, ioff=10):
    """ Generate list of datetimes from
    list of filenames
    
    Works for MODIS and VIIRS

    Args:
        filenames (list): List of filenames.
            Need to be the base (no path)
        ioff (int, optional): [description]. Defaults to 10.

    Returns:
        list:  List of datetimes
    """
    # Dates
    dtimes = [datetime.datetime(int(ifile[1+ioff:5+ioff]),
                                int(ifile[5+ioff:7+ioff]),
                                int(ifile[7 + ioff:9+ioff]),
                                int(ifile[10+ioff:12+ioff]),
                                int(ifile[12+ioff:14+ioff]))
                for ifile in filenames]
    # Return
    return dtimes
