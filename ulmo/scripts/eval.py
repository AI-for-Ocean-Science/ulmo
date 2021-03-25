"""
Simple script to run Evals on MODIS L2 data
"""

import os
import numpy as np

from ulmo.models import io as model_io
from ulmo import io as ulmo_io

from IPython import embed


def run_evals(years, flavor, clobber=False, local=False):
    """Main method to evaluate the model

    Outputs are written to hard-drive in a sub-folder
    named Evaluations/

    Args:
        years (tuple): (start year [int], end year [int])
        flavor (str): Model to apply.  ['std']
        clobber (bool, optional): Clobber existing outputs. Defaults to False.
        local (bool, optional): Load model and data locally. 
            Otherwise use s3 storage. Defaults to False.
        dataset (str, optional): Dataset to work on.  Defaults to 'MODIS_L2'

    Raises:
        IOError: [description]
    """

    # Load model
    pae = model_io.load_modis_l2(flavor=flavor, local=local)
    print("Model loaded!")

    # Allow for various datasets
    preproc_folder = 'PreProc'

    # Prep
    for year in years:
        # Input
        data_file = os.path.join(preproc_folder, 
                                 'MODIS_R2019_{}_95clear_128x128_preproc_{}.h5'.format(year, flavor))
        # Grab from s3 (faster local runnin)
        if not local:
            if not os.path.isdir('PreProc'):
                os.mkdir('PreProc')
            ulmo_io.s3.Bucket('modis-l2').download_file(data_file, data_file)
            print("Dowloaded: {} from s3".format(data_file))
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
        pae.eval_data_file(data_file, 'valid', 
                              log_prob_file, 
                              csv=False)  # Tends to crash on kuber

        # Remove local
        if not local:
            os.remove(data_file)


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
