""" Result methods"""

import glob, os
import numpy as np

import pandas
import datetime

from astropy.coordinates import SkyCoord
from astropy import units

from ulmo import defs

from IPython import embed


def load_log_prob(pproc, table_files=None, add_UID=False, feather=False):
    """
    Load log probabilities

    Parameters
    ----------
    pproc : str
        std or loggrad
    table_files

    Returns
    -------

    """

    if feather:
        ffile = os.path.join(defs.eval_path, 'R2010_results_{}.feather'.format(pproc))
        print("Loading: {}".format(ffile))
        return pandas.read_feather(ffile)

    # Load up the tables
    if table_files is None:
        table_files = glob.glob(os.path.join(defs.eval_path, 'R2010_on*{}_log_prob.csv'.format(pproc)))
    table_files.sort()


    ioff = 10
    evals_tbl = pandas.DataFrame()
    for table_file in table_files:
        print("Loading: {}".format(table_file))
        df = pandas.read_csv(table_file)
        # Dates
        dtimes = [datetime.datetime(int(ifile[1+ioff:5+ioff]),
                                    int(ifile[5+ioff:7+ioff]),
                                    int(ifile[7 + ioff:9+ioff]),
                                    int(ifile[10+ioff:12+ioff]),
                                    int(ifile[12+ioff:14+ioff]))
                  for ifile in df['filename'].values]

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


def random_imgs(evals_tbl, years, dyear, verbose=False):
    """
    Generate a set of random images

    Parameters
    ----------
    evals_tbl
    years
    dyear
    verbose

    Returns
    -------
    pandas.DataFrame

    """

    # Coords
    coords = SkyCoord(b=evals_tbl.latitude * units.deg,
                      l=evals_tbl.longitude * units.deg,
                      frame='galactic')

    # Loop time
    used_coords = None
    for_gallery = []
    for year in years:
        # Cut on date
        t0 = datetime.datetime(year, 1, 1)
        t1 = datetime.datetime(year + dyear, 1, 1)
        in_time = np.where((evals_tbl.date >= t0) & (evals_tbl.date < t1))[0]
        if verbose:
            print('Year {}:, n_options={}'.format(year, len(in_time)))
        # Grab one
        if used_coords is not None:
            # Ugly loop
            all_seps = np.zeros((len(used_coords), len(in_time)))
            for kk, ucoord in enumerate(used_coords):
                seps = ucoord.separation(coords[in_time])
                all_seps[kk, :] = seps.to('deg').value
            # Minimum for each
            min_seps = np.min(all_seps, axis=0)
            best = np.argmax(min_seps)
            for_gallery.append(in_time[best])
            used_coords = coords[np.array(for_gallery)]
        else:
            # Take a random one
            rani = np.random.randint(low=0, high=len(in_time), size=1)[0]
            for_gallery.append(in_time[rani])
            used_coords = coords[np.array(for_gallery)]

    # Return table of random choices
    return evals_tbl.iloc[for_gallery].copy()


def build_feather():
    """
    Generate a feather version of the pandas table

    Returns
    -------

    """
    # Standard
    for model in ['std', 'loggrad']:
        df = load_log_prob(model, add_UID=True)
        uid = df['UID'].values
        if len(uid) != len(np.unique(uid)):
            print("Not all indices are unique.  Boo hoo")
        df.reset_index().to_feather('R2010_results_{}.feather'.format(model))

# Command line execution
if __name__ == '__main__':
    build_feather()


