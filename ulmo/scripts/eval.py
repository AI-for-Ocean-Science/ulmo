"""
Simple script to run Evals
"""

import os
import numpy as np

from ulmo.ood import ood
from ulmo.models import io as model_io

from IPython import embed


def run_evals(years, flavor, clobber=False, local=False):

    # Load model
    pae = model_io.load_modis_l2(flavor=flavor, local=local)
    print("Model loaded!")

    # Prep
    for year in years:
        # Input
        data_file = 'PreProc/MODIS_R2019_{}_95clear_128x128_preproc_{}.h5'.format(year, flavor)
        if not local:
            data_file = 's3://modis-l2/'+data_file
        # Check
        if local:
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
    parser.add_argument("--clobber", default=False, action="store_true", help="Debug?")
    parser.add_argument("--local", default=False, action="store_true", help="Use local storage")

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

    run_evals(years, pargs.flavor, clobber=pargs.clobber, local=pargs.local)
