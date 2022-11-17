""" vet of the SST L3S dataset
"""
from email import header
from operator import mod
import os
from typing import IO
import numpy as np
import glob

import time
import h5py
import numpy as np
from tqdm.auto import trange
import argparse

import pandas
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils
from ulmo.preproc import utils as pp_utils

from ulmo.ssl.util import adjust_learning_rate
from ulmo.ssl.util import set_optimizer, save_model
from ulmo.ssl import latents_extraction
from ulmo.ssl import umap as ssl_umap

from ulmo.ssl.train_util import option_preprocess
from ulmo.ssl.train_util import modis_loader, set_model
from ulmo.ssl.train_util import train_model

from ulmo.preproc import io as pp_io 
from ulmo.modis import utils as modis_utils
from ulmo.modis import extract as modis_extract
from ulmo.analysis import evaluate as ulmo_evaluate 

from IPython import embed

debug_file = 's3://sst_l3s/Tables/SST_L3S_debug.parquet'
tbl_file = 's3://sst_l3s/Tables/SST_L3S.parquet'

def extract_files(debug=False, n_cores=20,
                  local=False):
    # Init
    if debug:
        tbl_file = debug_file

    # Grab all the files
    if local:
        mat_files = glob.glob(
            os.path.join(os.getenv('OS_DATA'),
                     'SST', 
                     'L3S', 
                     'daily', 
                     'extracted', 
                     '*',
                     '*.mat'))
    mat_files.sort()

    # Loop on years
    all_years = np.unique([os.path.basename(ifile)[0:4] for ifile in mat_files])

    if debug:
        all_years = all_years[0:1]

    # 
    for year in all_years:
        # Open table
        if local:
            tbl_file = os.path.join(os.getenv('OS_DATA'),
                     'SST', 
                     'L3S', 
                     'daily', 
                     'extracted', year,
                     f'cutout_table_for_{year}.parquet')
        tbl_year = pandas.read_parquet(tbl_file)
        # Output basename
        basename = f'SST_L3S_{year}_extract.h5'
        sst_a = []
        embed(header='85 of sst')
        for mat_file in mat_files:
            if year not in mat_file:
                continue
            # Load up .mat file
            f = h5py.File(mat_file, 'r')
            # Data
            sst_a.append(f['sst_a'][:])
            # Translate PC columns
            transl = dict(
                lat='cutout_latitude',
                lon='cutout_longitude',
                T10='cutout_T_10',
                T90='cutout_T_90',
                Tmin='cutout_T_min',
                Tmax='cutout_T_max',
                mean_temperature='cutout_mean_temp',
                clear_fraction='cutout_fraction_clear',
            )
            # Meta
            for key in ['lat', 'lon', 'col', 'row', 
                        'clear_fraction', 
                        'datetime', 
                        'mean_temperature', 'Tmin', 
                        'Tmax', 'T90', 'T10', 'filename', 
                        'field_size']:
                embed(header='60 of vet')

def preproc(debug=False, n_cores=20):
    """Pre-process the files

    Args:
        n_cores (int, optional): Number of cores to use
    """
    # Build the table

    # Pre-process 
    modis_tbl = pp_utils.preproc_tbl(modis_tbl, 1., 
                                     's3://modis-l2',
                                     preproc_root='standard',
                                     inpainted_mask=False,
                                     use_mask=True,
                                     debug=debug,
                                     remove_local=False,
                                     nsub_fields=10000,
                                     n_cores=n_cores)
    # Vet
    assert cat_utils.vet_main_table(modis_tbl)

    # Final write
    if not debug:
        ulmo_io.write_main_table(modis_tbl, tbl_20s_file)
    else:
        ulmo_io.write_main_table(modis_tbl, 'preproc_debug.parquet', to_s3=False)
        print('Wrote: preproc_debug.parquet')

def slurp_tables(debug=False, orig_strip=False):
    tbl_20s_file = 's3://modis-l2/Tables/MODIS_L2_20202021.parquet'
    full_tbl_file = 's3://modis-l2/Tables/MODIS_SSL_96clear.parquet'

    # Load
    modis_20s_tbl = ulmo_io.load_main_table(tbl_20s_file)

    # Check
    modis_20s_tbl['DT'] = modis_20s_tbl.T90 - modis_20s_tbl.T10

    def plot_DTvsLL(tbl):
        print("Generating the plot...")
        ymnx = [-5000., 1000.]
        jg = sns.jointplot(data=tbl, x='DT', y='LL', kind='hex',
                        bins='log', gridsize=250, xscale='log',
                        cmap=plt.get_cmap('autumn'), mincnt=1,
                        marginal_kws=dict(fill=False, color='black', 
                                            bins=100)) 
        jg.ax_joint.set_xlabel(r'$\Delta T$')
        jg.ax_joint.set_ylim(ymnx)
        plt.colorbar()
        plt.show()

    # New check
    print("New years")
    plot_DTvsLL(modis_20s_tbl)

    # Strip original if it is there..
    modis_full = ulmo_io.load_main_table(full_tbl_file)
    print("Full table")
    modis_full['DT'] = modis_full.T90 - modis_full.T10
    plot_DTvsLL(modis_full)
    if orig_strip:
        bad = modis_full.UID.values == modis_full.iloc[0].UID
        bad[0] = False
    else:
        bad2020 = np.array(['R2019_2020' in pp_file for pp_file in modis_full.pp_file.values])
        bad2021 = np.array(['R2019_2021' in pp_file for pp_file in modis_full.pp_file.values])
        bad = bad2020 | bad2021
    modis_full = modis_full[~bad].copy()

    # Another check

    # Rename ulmo_pp_type
    modis_20s_tbl.rename(columns={'pp_type':'ulmo_pp_type'}, inplace=True)

    # Deal with filenames
    filenames = []
    for ifile in modis_20s_tbl.filename:
        filenames.append(os.path.basename(ifile))
    modis_20s_tbl['filename'] = filenames


    # Fill up the new table with dummy values
    for key in modis_full.keys():
        if key not in modis_20s_tbl.keys():
            modis_20s_tbl[key] = modis_full[key].values[0]

    # Generate new UIDs
    modis_20s_tbl['UID'] = modis_utils.modis_uid(modis_20s_tbl)

    # Drop unwanted
    for key in modis_20s_tbl.keys():
        if key not in modis_full.keys():
            modis_20s_tbl.drop(key, axis=1, inplace=True)

    # Cut on 96% clear
    cut = modis_20s_tbl.clear_fraction < 0.04
    modis_20s_tbl = modis_20s_tbl[cut].copy()

    # Concat
    modis_full = pandas.concat([modis_full, modis_20s_tbl],
                               ignore_index=True)
    modis_full.drop(columns='DT', inplace=True)

    if debug:
        embed(header='672 of v4')

    # Vet
    assert cat_utils.vet_main_table(modis_full, cut_prefix='ulmo_')

    # Final write
    if not debug:
        ulmo_io.write_main_table(modis_full, full_tbl_file)



# DEPRECATED
#def cut_96(debug=False):
#    """ Cut to 96% clear 
#    """
#    full_tbl_file = 's3://modis-l2/Tables/MODIS_SSL_96clear.parquet'
#
#    # Load
#    modis_full = ulmo_io.load_main_table(full_tbl_file)
#
#
#    # Cut
#    cut = modis_full.clear_fraction < 0.04
#    modis_full = modis_full[cut].copy()
#
#    # Vet
#    assert cat_utils.vet_main_table(modis_full, cut_prefix='ulmo_')
#
#    # Final write
#    if not debug:
#        ulmo_io.write_main_table(modis_full, full_tbl_file)

def modis_ulmo_evaluate(debug=False):
    """ Run Ulmo on the 2020s data

    Args:
        debug (bool, optional): _description_. Defaults to False.
    """

    # Load 2020s
    tbl_20s_file = 's3://modis-l2/Tables/MODIS_L2_20202021.parquet'
    modis_tbl = ulmo_io.load_main_table(tbl_20s_file)

    # Deal with pp_filenames
    if 'standardT' in modis_tbl.pp_file.values[0]:
        pp_filenames = []
        for ifile in modis_tbl.pp_file:
            pp_filenames.append(os.path.basename(ifile.replace('standardT', 'std')))
        modis_tbl['pp_file'] = pp_filenames

    if debug:
        embed(header='687 of v4')

    # Evaluate
    modis_tbl = ulmo_evaluate.eval_from_main(modis_tbl)

    # Write 
    assert cat_utils.vet_main_table(modis_tbl)

    if not debug:
        ulmo_io.write_main_table(modis_tbl, tbl_20s_file)


def calc_dt40(opt_path, debug:bool=False, local:bool=False,
              redo=False):
    """ Calculate DT40 in all the 96 clear data

    Args:
        opt_path (str): Path to the options file 
        debug (bool, optional): _description_. Defaults to False.
        local (bool, optional): _description_. Defaults to False.
    """
    # Options (for the Table name)
    opt = option_preprocess(ulmo_io.Params(opt_path))

    # Tables
    if redo:
        tbl_file = 's3://modis-l2/Tables/MODIS_SSL_v4.parquet'
        modis_tbl = ulmo_io.load_main_table(tbl_file)
    else:
        if not debug:
            tbl_file = 's3://modis-l2/Tables/MODIS_SSL_96clear.parquet'
        else:
            tbl_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2', 'Tables', 
                                    'MODIS_SSL_96clear.parquet')
        modis_tbl = ulmo_io.load_main_table(tbl_file)
        modis_tbl['DT40'] = 0.
        if debug:
            full_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2', 'Tables', 
                                    'MODIS_L2_std.parquet')
        else:
            full_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
        full_modis_tbl = ulmo_io.load_main_table(full_file)

        # Fix ULMO crap and more
        print("Fixing the ulmo crap")
        ulmo_pp_idx = modis_tbl.pp_idx.values
        ulmo_pp_type = modis_tbl.pp_type.values
        ulmo_pp_file = modis_tbl.pp_file.values
        mtch = cat_utils.match_ids(modis_tbl.UID, full_modis_tbl.UID, 
                                require_in_match=False) # 2020, 2021
        new = mtch >= 0
        ulmo_pp_idx[new] = full_modis_tbl.pp_idx.values[mtch[new]]
        ulmo_pp_type[new] = full_modis_tbl.pp_type.values[mtch[new]]
        ulmo_pp_file[new] = full_modis_tbl.pp_file.values[mtch[new]]
        modis_tbl['ulmo_pp_idx'] = ulmo_pp_idx
        modis_tbl['ulmo_pp_type'] = ulmo_pp_type
        modis_tbl['ulmo_pp_file'] = ulmo_pp_file
        print("Done..")

    # Fix s3 in 2020
    new_pp_files = []
    for pp_file in modis_tbl.pp_file:
        if 's3' not in pp_file:
            ipp_file = 's3://modis-l2/PreProc/'+pp_file
        else:
            ipp_file = pp_file
        # Standard
        if 'standard' in ipp_file:
            ipp_file = ipp_file.replace('standard', 'std')
        new_pp_files.append(ipp_file)
            
    modis_tbl['pp_file'] = new_pp_files
    
    # Grab the list
    preproc_files = np.unique(modis_tbl.pp_file.values)
    if redo:
        # Only 2020, 2021
        preproc_files = preproc_files[-2:]

    # Loop on files
    for pfile in preproc_files:
        #if debug and '2010' not in pfile:
        #    continue
        basename = os.path.basename(pfile)
        if local:
            basename = os.path.join(os.getenv('SST_OOD'),
                                'MODIS_L2', 'PreProc', basename) 
        else:
            # Download?
            if not os.path.isfile(basename):
                ulmo_io.download_file_from_s3(basename, pfile)

        # Open me
        print(f"Starting on: {basename}")
        f = h5py.File(basename, 'r')

        # Load it all
        DT40(f, modis_tbl, pfile, itype='valid', verbose=debug)
        if 'train' in f.keys():
            DT40(f, modis_tbl, pfile, itype='train', verbose=debug)

        # Close
        f.close()

        # Check
        if debug:
            embed(header='725 of v4')

        # Remove 
        if not debug and not local:
            os.remove(basename)
    # Vet
    assert cat_utils.vet_main_table(modis_tbl, cut_prefix='ulmo_')

    # Save
    if not debug:
        ulmo_io.write_main_table(modis_tbl, opt.tbl_file)
    else:
        embed(header='740 of v4')

    print("All done")

def DT40(f:h5py.File, modis_tbl:pandas.DataFrame, 
         pfile:str, itype:str='train', verbose=False):
    """Calculate DT40 for a given file

    Args:
        f (h5py.File): _description_
        modis_tbl (pandas.DataFrame): _description_
        pfile (str): _description_
        itype (str, optional): _description_. Defaults to 'train'.
    """
    fields = f[itype][:]
    if verbose:
        print("Calculating T90")
    T_90 = np.percentile(fields[:, 0, 32-20:32+20, 32-20:32+20], 
        90., axis=(1,2))
    if verbose:
        print("Calculating T10")
    T_10 = np.percentile(fields[:, 0, 32-20:32+20, 32-20:32+20], 
        10., axis=(1,2))
    DT_40 = T_90 - T_10
    # Fill
    ppt = 0 if itype == 'valid' else 1
    idx = (modis_tbl.pp_file == pfile) & (modis_tbl.ulmo_pp_type == ppt)
    pp_idx = modis_tbl[idx].ulmo_pp_idx.values
    modis_tbl.loc[idx, 'DT40'] = DT_40[pp_idx]
    return 

def ssl_v4_umap(opt_path:str, debug=False, local=False):
    """Run a UMAP analysis on all the MODIS L2 data
    v4 model

    Either 2 or 3 dimensions

    Args:
        model_name: (str) model name 
        ntrain (int, optional): Number of random latent vectors to use to train the UMAP model
        debug (bool, optional): For testing and debuggin 
        ndim (int, optional): Number of dimensions for the embedding
    """
    # Load up the options file
    opt = option_preprocess(ulmo_io.Params(opt_path))

    # Load v4 Table
    if local:
        tbl_file = os.path.join(os.getenv('SST_OOD'),
                                'MODIS_L2', 'Tables', 
                                os.path.basename(opt.tbl_file))
    else:                            
        tbl_file = opt.tbl_file
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Base
    base1 = '96clear_v4'

    #for subset in ['DTall']:
    #for subset in ['DT5']:
    for subset in ['DT15', 'DTall', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5']:
        # Files
        outfile = os.path.join(
            os.getenv('SST_OOD'), 
            f'MODIS_L2/Tables/MODIS_SSL_{base1}_{subset}.parquet')
        umap_savefile = os.path.join(
            os.getenv('SST_OOD'), 
            f'MODIS_L2/UMAP/MODIS_SSL_{base1}_{subset}_UMAP.pkl')

        # DT cut
        DT_cut = None if subset == 'DTall' else subset

        if debug:
            embed(header='786 of v4')

        # Run
        ssl_umap.umap_subset(modis_tbl.copy(),
                             opt_path, 
                             outfile, 
                             local=local,
                             DT_cut=DT_cut, debug=debug, 
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
                        default='opts_ssl_modis_v4.json',
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

    # python sst_l3s_vet.py --func_flag extract --debug --local
    if args.func_flag == 'extract':
        extract_files(debug=args.debug, 
                      local=args.local)
    
    # python ssl_modis_v4.py --func_flag preproc --debug
    if args.func_flag == 'preproc':
        modis_20s_preproc(debug=args.debug)

    # python ssl_modis_v4.py --func_flag ulmo_evaluate --debug
    #  This comes before the slurp and cut
    if args.func_flag == 'ulmo_evaluate':
        modis_ulmo_evaluate(debug=args.debug)

    # python ssl_modis_v4.py --func_flag slurp_tables --debug
    if args.func_flag == 'slurp_tables':
        slurp_tables(debug=args.debug)

    # python ssl_modis_v4.py --func_flag cut_96 --debug
    #if args.func_flag == 'cut_96':
    #    cut_96(debug=args.debug)

    # python ssl_modis_v4.py --func_flag ssl_evaluate --debug
    if args.func_flag == 'ssl_evaluate':
        main_ssl_evaluate(args.opt_path, debug=args.debug)
        
    # python ssl_modis_v4.py --func_flag DT40 --debug --local
    if args.func_flag == 'DT40':
        calc_dt40(args.opt_path, debug=args.debug, local=args.local,
                  redo=args.redo)

    # python ssl_modis_v4.py --func_flag umap --debug --local
    if args.func_flag == 'umap':
        ssl_v4_umap(args.opt_path, debug=args.debug, local=args.local)
