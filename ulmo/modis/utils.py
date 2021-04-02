import datetime

def times_from_filenames(filenames, ioff=10):
    # Dates
    dtimes = [datetime.datetime(int(ifile[1+ioff:5+ioff]),
                                int(ifile[5+ioff:7+ioff]),
                                int(ifile[7 + ioff:9+ioff]),
                                int(ifile[10+ioff:12+ioff]),
                                int(ifile[12+ioff:14+ioff]))
                for ifile in filenames]
    # Return
    return dtimes
