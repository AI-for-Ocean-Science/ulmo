""" Figures for SSL paper on MODIS """
import os, sys
from typing import IO
import numpy as np
import scipy
from scipy import stats
from urllib.parse import urlparse

import argparse

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import h5py

from ulmo import plotting
from ulmo.utils import utils as utils

from ulmo import io as ulmo_io
from ulmo.ssl import single_image as ssl_simage
from ulmo.utils import image_utils

from IPython import embed

# Plot ranges for the UMAP
xrngs_CF = -5., 10.
yrngs_CF = 4., 14.
xrngs_CF_DT0 = -0.5, 8.5
yrngs_CF_DT0 = 3, 12.
xrngs_CF_DT1 = -0.5, 10.
yrngs_CF_DT1 = -1.5, 8.
xrngs_CF_DT15 = 0., 10.
yrngs_CF_DT15 = -1., 7.
xrngs_CF_DT2 = 0., 11.5
yrngs_CF_DT2 = -1., 8.
xrngs_95 = -4.5, 8.
yrngs_95 = 4.5, 10.5

# U3
xrngs_CF_U3 = -4.5, 7.5
yrngs_CF_U3 = 6.0, 13.
xrngs_CF_U3_12 = yrngs_CF_U3
yrngs_CF_U3_12 = 5.5, 13.5

metric_lbls = dict(min_slope=r'$\alpha_{\rm min}$',
                   clear_fraction='1-CC',
                   DT=r'$\Delta T$',
                   lowDT=r'$\Delta T_{\rm low}$',
                   absDT=r'$|T_{90}| - |T_{10}|$',
                   LL='LL',
                   zonal_slope=r'$\alpha_{\rm AS}}$',
                   merid_slope=r'$\alpha_{\rm AT}}$',
                   )


# Local
sys.path.append(os.path.abspath("../Analysis/py"))
#import ssl_paper_analy


def fig_LLvsDT(clear_frac, outroot='fig_viirs_LLvsDT', 
               local=True, vmax=None, umap_dim=2,
                table=None, cmap=None, cuts=None, scl = 1, debug=False):
    """ Bivariate of LL vs. DT

    Args:
        outfile (str, optional): [description]. Defaults to 'fig_LLvsDT.png'.
        local (bool, optional): [description]. Defaults to False.
        vmax ([type], optional): [description]. Defaults to None.
        cmap ([type], optional): [description]. Defaults to None.
        cuts ([type], optional): [description]. Defaults to None.
        scl (int, optional): [description]. Defaults to 1.
        debug (bool, optional): [description]. Defaults to False.
    """

    # Load table
    icc = int(100*clear_frac)
    tbl_base = 'VIIRS_all_95clear_std.parquet'
    tbl_file = os.path.join(os.getenv('SST_OOD'), 'VIIRS', 'Tables', tbl_base)
    viirs_tbl = ulmo_io.load_main_table(tbl_file)

    # DT
    viirs_tbl['DT'] = viirs_tbl.T90 - viirs_tbl.T10

    # Cuts
    gd_cc = viirs_tbl.clear_fraction < (1-clear_frac)
    gd_LL = viirs_tbl.LL > -5000.

    viirs_tbl = viirs_tbl[gd_cc & gd_LL].copy()

    # Random cut (speed things up a bit)
    if len(viirs_tbl) > 2000000:
        idx = np.random.choice(len(viirs_tbl), 2000000, replace=False)
        viirs_tbl = viirs_tbl.iloc[idx].copy()
    viirs_tbl = viirs_tbl.reset_index()
    print(f"Plotting: {len(viirs_tbl)} cutouts")

    outfile = outroot+f'_{icc}.png'

    # Debug?

    # Plot
    fig = plt.figure()#figsize=(9, 12))
    plt.clf()

    ymnx = [-5000., 1000.]

    print("Starting plot...")
    jg = sns.jointplot(data=viirs_tbl, x='DT', y='LL', kind='hex',
                       bins='log', 
                       gridsize=250, 
                       xscale='log',
                       cmap=plt.get_cmap('autumn'), mincnt=1,
                       marginal_kws=dict(fill=False, color='black', 
                                         bins=100)) 
    # Axes                                 
    jg.ax_joint.set_xlabel(r'$\Delta T$')
    jg.ax_joint.set_ylim(ymnx)
    jg.fig.set_figwidth(8.)
    jg.fig.set_figheight(7.)

    plotting.set_fontsize(jg.ax_joint, 16.)

    # Save
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()
    print('Wrote {:s}'.format(outfile))


        
#### ########################## #########################
def main(pargs):

    # LL vs DT
    if pargs.figure == 'LLvsDT':
        fig_LLvsDT(pargs.cc, local=pargs.local, debug=pargs.debug)

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("VIIRS figs")
    parser.add_argument("figure", type=str, 
                        help="function to execute: 'LLvsDT'")
    parser.add_argument("--cc", type=float, help="Clear fraction, e.g. 0.99")
    parser.add_argument('--local', default=False, action='store_true',
                        help='Local?')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    args = parser.parse_args()
    
    return args

# Command line execution
if __name__ == '__main__':

    pargs = parse_option()
    main(pargs)

# LL vs DT -- python py/viirs.py LLvsDT --cc 0.95 --local