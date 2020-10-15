""" Result methods"""

import glob, os
import numpy as np

import pandas
import datetime

from astropy.coordinates import SkyCoord
from astropy import units

from ulmo import defs

from IPython import embed

def load_log_prob(pproc, table_files=None):

    # Load up the tables
    if table_files is None:
        table_files = glob.glob(os.path.join(defs.eval_path, 'R2010_on*{}_log_prob.csv'.format(pproc)))
    table_files.sort()

    # Cut down?
    # table_files = table_files[0:2]

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
        #except:
        #    embed(header='32 of results')
        df['date'] = dtimes
        evals_tbl = pandas.concat([evals_tbl, df])

    #print('NEED TO ADD IN 2010!!!')

    # Return
    return evals_tbl


def random_imgs(evals_tbl, years, dyear, top=1000, verbose=False):
    # Cut
    isrt = np.argsort(evals_tbl.log_likelihood)
    topN = evals_tbl.iloc[isrt[0:top]]

    # Coords
    coords = SkyCoord(b=topN.latitude * units.deg, l=topN.longitude * units.deg, frame='galactic')

    # Loop time
    used_coords = None
    for_gallery = []
    for year in years:
        # Cut on date
        t0 = datetime.datetime(year, 1, 1)
        t1 = datetime.datetime(year + dyear, 1, 1)
        in_time = np.where((topN.date >= t0) & (topN.date < t1))[0]
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
    return topN.iloc[for_gallery]