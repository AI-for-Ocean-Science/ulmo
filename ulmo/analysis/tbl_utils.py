""" Utility functions related to tables for analysis """

import glob, os
import numpy as np

import pandas
import datetime

from ulmo import defs

from IPython import embed

def load_log_prob(pproc, table_files=None, add_UID=False, feather=False,
                  date_format='YYYYMMDDHHMM'):
    """
    Load log probabilities

    Parameters
    ----------
    pproc : str
        std or loggrad or None
    table_files : list, optional
        List of table CSV files
    feather : bool, optional

    Returns
    -------

    table : pandas.DataFrame

    """

    if feather:
        ffile = os.path.join(defs.eval_path, 
                             'R2010_results_{}.feather'.format(pproc))
        print("Loading: {}".format(ffile))
        return pandas.read_feather(ffile)

    # Load up the tables
    if table_files is None:
        table_files = glob.glob(os.path.join(defs.eval_path, 'R2010_on*{}_log_prob.csv'.format(pproc)))
    table_files.sort()


    evals_tbl = pandas.DataFrame()
    for table_file in table_files:
        print("Loading: {}".format(table_file))
        df = pandas.read_csv(table_file)
        # Dates
        if date_format == 'YYYYMMDDHHMM':
            ioff = 10
            dtimes = [datetime.datetime(int(ifile[1+ioff:5+ioff]),
                                    int(ifile[5+ioff:7+ioff]),
                                    int(ifile[7 + ioff:9+ioff]),
                                    int(ifile[10+ioff:12+ioff]),
                                    int(ifile[12+ioff:14+ioff]))
                  for ifile in df['filename'].values]
        elif date_format == 'YYYYDDDHHMMSS':
            ioff = 0
            dtimes = [datetime.datetime(int(ifile[1+ioff:5+ioff]), 1, 1) + datetime.timedelta(
                days=int(ifile[5+ioff:8+ioff])-1,
                hours=int(ifile[8 + ioff:10+ioff]),
                minutes=int(ifile[10+ioff:12+ioff]),
                seconds= int(ifile[12+ioff:14+ioff]))
                  for ifile in df['filename'].values]
        else:
            raise IOError("Bad date_format")

        df['date'] = dtimes
        # Unique identifier
        if add_UID:
            tlong = df['date'].values.astype(np.int64) // 10000000000
            lats = np.round((df.latitude.values + 90)*10000).astype(int)
            lons = np.round((df.longitude.values + 180)*100000).astype(int)
            uid = [np.int64('{:s}{:d}{:d}'.format(str(t)[:-5],lat,lon))
                   for t,lat,lon in zip(tlong, lats, lons)]
            if len(uid) != len(np.unique(uid)):
                embed(header='67 of results')
            df['UID'] = np.array(uid).astype(np.int64)
        evals_tbl = pandas.concat([evals_tbl, df])

    # Return
    return evals_tbl