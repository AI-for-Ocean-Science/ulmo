""" Nenya Analayis of MODIS -- 
96% clear 
New UMAP
"""
import os
import numpy as np

import numpy as np
import argparse

from ulmo import io as ulmo_io

from ulmo.nenya import nenya_umap
from ulmo.nenya import defs as nenya_defs

from ulmo.nenya.train_util import option_preprocess


from IPython import embed



def nenya_v5_umap(opt_path:str, debug=False, local=False, metric:str='DT40'):
    """Run a UMAP analysis on all the MODIS L2 data
    v5 model

    2 dimensions

    Args:
        model_name: (str) model name 
        ntrain (int, optional): Number of random latent vectors to use to train the UMAP model
        debug (bool, optional): For testing and debuggin 
        ndim (int, optional): Number of dimensions for the embedding
    """
    # Load up the options file
    opt = option_preprocess(ulmo_io.Params(opt_path))

    # Load v5 Table
    if local:
        tbl_file = os.path.join(os.getenv('OS_SST'),
                                'MODIS_L2', 'Tables', 
                                os.path.basename(opt.tbl_file))
    else:                            
        tbl_file = opt.tbl_file
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Add slope
    modis_tbl['min_slope'] = np.minimum(
        modis_tbl.zonal_slope, modis_tbl.merid_slope)

    # Base
    base1 = '96clear_v5'

    if 'DT' in metric: 
        subsets =  ['DT15', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5', 'DTall']
        if debug:
            subsets = ['DT5']
    elif metric == 'alpha':
        subsets = list(nenya_defs.umap_alpha.keys())
        if debug:
            subsets = ['a0']
    else:
        raise ValueError("Bad metric")

    # Loop me
    for subset in subsets:
        # Files
        outfile = os.path.join(
            os.getenv('OS_SST'), 
            f'MODIS_L2/Nenya/Tables/MODIS_SSL_{base1}_{subset}.parquet')
        umap_savefile = os.path.join(
            os.getenv('OS_SST'), 
            f'MODIS_L2/Nenya/UMAP/MODIS_SSL_{base1}_{subset}_UMAP.pkl')

        DT_cut = None 
        alpha_cut = None 
        if 'DT' in metric:
            # DT cut
            DT_cut = None if subset == 'DTall' else subset
        elif metric == 'alpha':
            alpha_cut = subset
        else:
            raise ValueError("Bad metric")

        if debug:
            embed(header='940 of v5')

        # Run
        if os.path.isfile(umap_savefile):
            print(f"Skipping UMAP training as {umap_savefile} already exists")
            train_umap = False
        else:
            train_umap = True
        # Can't do both so quick check
        if DT_cut is not None and alpha_cut is not None:
            raise ValueError("Can't do both DT and alpha cuts")

        # Do it
        nenya_umap.umap_subset(modis_tbl.copy(),
                             opt_path, 
                             outfile, 
                             local=local,
                             DT_cut=DT_cut, 
                             alpha_cut=alpha_cut, 
                             debug=debug, 
                             train_umap=train_umap, 
                             umap_savefile=umap_savefile,
                             remove=False, CF=False)

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("argument for training.")
    parser.add_argument("--opt_path", type=str, 
                        default='opts_nenya_modis_v5.json',
                        help="Path to options file")
    parser.add_argument("--func_flag", type=str, 
                        help="flag of the function to be execute: train,evaluate,umap,umap_ndim3,sub2010,collect")
    parser.add_argument("--model", type=str, 
                        default='2010', help="Short name of the model used [2010,CF]")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    parser.add_argument('--local', default=False, action='store_true',
                        help='Local?')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Clobber existing files')
    parser.add_argument('--redo', default=False, action='store_true',
                        help='Redo?')
    parser.add_argument("--outfile", type=str, 
                        help="Path to output file")
    parser.add_argument("--umap_file", type=str, 
                        help="Path to UMAP pickle file for analysis")
    parser.add_argument("--table_file", type=str, 
                        help="Path to Table file")
    parser.add_argument("--ncpu", type=int, help="Number of CPUs")
    parser.add_argument("--years", type=str, help="Years to analyze")
    parser.add_argument("--cf", type=float, 
                        help="Clear fraction (e.g. 96)")
    args = parser.parse_args()
    
    return args

        
if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    
    # python nenya_modis_v5.py --func_flag umap --debug --local
    if args.func_flag == 'umap':
        nenya_v5_umap(args.opt_path, debug=args.debug, local=args.local)
