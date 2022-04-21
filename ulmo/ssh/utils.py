import os
import datetime

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
