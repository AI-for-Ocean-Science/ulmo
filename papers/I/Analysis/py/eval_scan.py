"""
Simple script to run Evals
"""

import os
import numpy as np

from ulmo.ood import ood

from IPython import embed


def run_evals(flavor='std', clobber=False):

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

    # Input
    data_file = 'Scan/LL_map_preproc.h5'
    # Check
    if not os.path.isfile(data_file):
        raise IOError("This data file does not exist! {}".format(data_file))

    # Output
    log_prob_file = 'Scan/LL_map_log_prob.h5'
    if os.path.isfile(log_prob_file) and not clobber:
        print("Eval file {} exists! Skipping..".format(log_prob_file))
        return

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
    run_evals()


# Command line execution
if __name__ == '__main__':

    main()

