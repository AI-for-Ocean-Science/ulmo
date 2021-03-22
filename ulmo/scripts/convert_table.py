"""
Simple script to run Evals on MODIS L2 data
"""

import os

from ulmo import io as ulmo_io

from IPython import embed



def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Preproc images in an H5 file.')
    parser.add_argument("in_table", type=str, help="Name of input table file")
    parser.add_argument("out_table", type=str, help="Name of output table file [.csv, .feather, .parquet]")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs



def main(pargs):
    """ Run
    """
    import warnings

    # Load table
    main_tbl = ulmo_io.load_main_table(pargs.in_table)

    # Write
    if pargs.out_table[0:2] == 's3':
        to_s3=True
    else:
        to_s3=False
    ulmo_io.write_main_table(main_tbl, pargs.out_table, to_s3=to_s3)
    