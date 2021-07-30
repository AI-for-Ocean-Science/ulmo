""" Module for LLC Ulmo analyses"""
import os
import numpy as np

import pandas

from ulmo.llc import extract 
from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.utils import catalog as cat_utils


from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky

from IPython import embed

def parse_option():
    """
    This is a function used to parse the arguments in the training.

    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("VIIRS")
    parser.add_argument("--task", type=str,
                        help="task to execute: 'download', 'extract', 'preproc', 'eval'.")
    parser.add_argument("--orig_tbl_file", type=str,
                        help="Table file to start from")
    parser.add_argument("--new_tbl_file", type=str,
                        help="Table file to write to (if None, use the orig_tbl_file")
    parser.add_argument("--model", type=str,
                        help="s3 path to Ulmo model")
    args = parser.parse_args()

    return args



def ulmo_eval(pargs, noise=False, rename=True):

    # Load
    llc_table = ulmo_io.load_main_table(pargs.orig_tbl_file)

    # Evaluate
    llc_table = ulmo_evaluate.eval_from_main(llc_table)

    # Write 
    assert cat_utils.vet_main_table(llc_table, cut_prefix='modis_')
    outfile = pargs.orig_tbl_file if pargs.new_tbl_file is None else pargs.new_tbl_file
    ulmo_io.write_main_table(llc_table, outfile)


# Command line execution
if __name__ == '__main__':
    import sys

    # get the argument of training.
    pargs = parse_option()

    if pargs.task == 'eval':
        print("Evaluation Starts.")
        ulmo_eval(pargs)
        print("Evaluation Ends.")
    else:
        raise IOError("Bad choice")