""" Script to Figure out where a Cutout lies in the UMAP space(s) """
import os
import numpy as np
import argparse

from ulmo import io as ulmo_io

import ssl_defs
import ssl_paper_analy

from IPython import embed

def main(pargs):
    # Load table
    modis_tbl_file = os.path.join(os.getenv("SST_OOD"), 
                        'MODIS_L2', 'Tables', 
                        'MODIS_SSL_96clear_DT15.parquet') # This should be all
    modis_tbl = ulmo_io.load_main_table(modis_tbl_file)

    # Grab the cutout
    idx = np.where(modis_tbl.UID == pargs.UID)[0][0]
    cutout = modis_tbl.iloc[idx]

    # DT
    DT = cutout.T90 - cutout.T10
    subset = ssl_paper_analy.grab_subset(DT)

    # Open that table now!
    modis_subtbl_file = os.path.join(os.getenv("SST_OOD"), 
                        'MODIS_L2', 'Tables', 
                        f'MODIS_SSL_96clear_{subset}.parquet') # This should be all
    modis_subtbl = ulmo_io.load_main_table(modis_subtbl_file)
    idx = np.where(modis_subtbl.UID == pargs.UID)[0][0]
    cutout = modis_subtbl.iloc[idx]

    print(cutout)

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("Find the UMAP location of a cutout")
    parser.add_argument("UID", type=int, help="Cutout UID")
    parser.add_argument('--vmnx', default='-1,1', type=str, help="Color bar scale")
    parser.add_argument('--outfile', type=str, help="Outfile")
    parser.add_argument('--distr', type=str, default='normal',
                        help='Distribution to fit [normal, lognorm]')
    parser.add_argument('--local', default=False, action='store_true', 
                        help='Use local file(s)?')
    parser.add_argument('--table', type=str, default='std', 
                        help='Table to load: [std, CF, CF_DT2')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    args = parser.parse_args()
    
    return args

# Command line execution
if __name__ == '__main__':

    pargs = parse_option()
    main(pargs)

# python py/umap_cutout.py 1567148589832463570