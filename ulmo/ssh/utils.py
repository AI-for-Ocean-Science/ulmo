import os
import datetime

def times_from_filenames(filenames:list):#, ioff=10, toff=1):
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
    #dtimes = [datetime.datetime(int(float(ifile[1+ioff:5+ioff])), # Year
    #                            int(float(ifile[5+ioff:7+ioff])), # Month
    #                            int(float(ifile[7 + ioff:9+ioff])), # Day
    #                            int(float(ifile[9+ioff+toff:11+ioff+toff])), # Hour
    #                            int(float(ifile[11+ioff+toff:13+ioff+toff]))) # Minut
    
    dtimes = [datetime.datetime(int(ifile[16:20]), # Year
                                int(ifile[20:22]), # Month
                                int(ifile[22:24]), # Day
                                int(ifile[24:26]))#, # Hour
                                #int(float(ifile[:]))) # Minut
    
                for ifile in filenames]
    # Return
    return dtimes


fn = ['ssh_grids_v1812_1992100212','ssh_grids_v1812_1992100212','ssh_grids_v1812_1992100212']
test = times_from_filenames(fn)
print(test)


#ssh_grids_v1812_1992100212
#year:1992, month:10, day:02, hour:12(always)