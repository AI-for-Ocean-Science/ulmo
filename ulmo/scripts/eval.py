"""
Simple script to run Evals
"""

import os
import numpy as np

from ulmo.ood import ood

from IPython import embed


def run_evals(years, flavor, clobber=False):

    # Load model
    if flavor == 'loggrad':
        datadir = './Models/R2019_2010_128x128_loggrad'
        filepath = 'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_loggrad.h5'
    elif flavor == 'std':
        datadir = './Models/R2019_2010_128x128_std'
        filepath = 'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'
    pae = ood.ProbabilisticAutoencoder.from_json(datadir + '/model.json',
                                                 datadir=datadir,
                                                 filepath=filepath,
                                                 logdir=datadir)
    pae.load_autoencoder()
    pae.load_flow()

    print("Model loaded!")

    # Prep
    for year in years:
        # Input
        data_file = 'PreProc/MODIS_R2019_{}_95clear_128x128_preproc_{}.h5'.format(year, flavor)
        # Check
        if not os.path.isfile(data_file):
            raise IOError("This data file does not exist! {}".format(data_file))

        # Output
        log_prob_file = 'Evaluations/R2010_on_{}_95clear_128x128_preproc_{}_log_prob.h5'.format(year, flavor)
        if os.path.isfile(log_prob_file) and not clobber:
            print("Eval file {} exists! Skipping..".format(log_prob_file))
            continue

        # Run
        pae.compute_log_probs(data_file, 'valid', log_prob_file, csv=True)


def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Preproc images in an H5 file.')
    parser.add_argument("years", type=str, help="Begin, end year:  e.g. 2010,2012")
    parser.add_argument("flavor", type=str, help="Model (std, loggrad)")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs



def main(pargs):
    """ Run
    """
    import warnings

    # Generate year list
    year0, year1 = [int(year) for year in pargs.years.split(',')]
    years = np.arange(year0, year1+1).astype(int)

    run_evals(years, pargs.flavor)
