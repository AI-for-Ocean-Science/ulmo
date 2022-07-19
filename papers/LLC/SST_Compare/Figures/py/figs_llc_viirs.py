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

import cartopy.crs as ccrs
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

#llc_table = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
s3_llc_table_file = 's3://llc/Tables/llc_viirs_match.parquet'
s3_viirs_table_file = 's3://viirs/Tables/VIIRS_all_98clear_std.parquet'

#sys.path.append(os.path.abspath("../Analysis/py"))
#import ssl_paper_analy

def load_table(dataset, local=False, cut_lat=57.):
    if dataset == 'llc':
        if local:
            tbl_file = os.path.join(os.getenv('SST_OOD'),
                'LLC', 'Tables', os.path.basename(s3_llc_table_file))
        else:
            tbl_file = s3_llc_table_file
    elif dataset == 'viirs':
        if local:
            tbl_file = os.path.join(os.getenv('SST_OOD'),
                'VIIRS', 'Tables', os.path.basename(s3_viirs_table_file))
        else:
            tbl_file = s3_viirs_table_file
    # Load
    tbl = ulmo_io.load_main_table(tbl_file)

    # Cut?
    if cut_lat is not None:
        tbl = tbl[tbl.lat < cut_lat].copy()
        tbl.reset_index(drop=True, inplace=True)

    # Return
    return tbl

def fig_LL_histograms(outfile='fig_LL_histograms.png', local=True):

    llc_tbl = load_table('llc', local=local)
    viirs_tbl = load_table('viirs', local=local)
    print(f"N VIIRS: {len(viirs_tbl)}")

    xmnx = (-1000., 1200.)

    # Cut
    llc_tbl = llc_tbl[llc_tbl.LL > xmnx[0]].copy()
    viirs_tbl = viirs_tbl[viirs_tbl.LL > xmnx[0]].copy()

    # Figure time
    fig = plt.figure(figsize=(7, 4))
    plt.clf()
    ax = plt.gca()

    _ = sns.histplot(llc_tbl, x='LL', ax=ax, label='LLC')
    _ = sns.histplot(viirs_tbl, x='LL', ax=ax, color='orange', label='VIIRS')

    ax.set_xlim(xmnx)

    ax.legend()

    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_med_LL_head_tail(outfile='med_LL_diff_head_vs_tail.png',
                         hp_root='_v98'):

    # Load up the data
    evts_head    = np.load('../Analysis/evts_head'+hp_root, allow_pickle=True)
    hp_lons_tail = np.load('../Analysis/hp_lons_tail'+hp_root, allow_pickle=True)
    hp_lats_tail = np.load('../Analysis/hp_lats_tail'+hp_root, allow_pickle=True)
    meds_head = np.load('../Analysis/meds_head'+hp_root, allow_pickle=True)
    meds_tail = np.load('../Analysis/meds_tail'+hp_root, allow_pickle=True)
    
    fig = plt.figure(figsize=(12,8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm = plt.get_cmap('coolwarm')
    # Cut
    good = np.invert(evts_head.mask)
    img = plt.scatter(x=hp_lons_tail[good],
        y=hp_lats_tail[good],
        c=meds_head[good]- meds_tail[good], vmin = -300, vmax = 300, 
        cmap=cm,
        s=1,
        transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    clbl = r'$LL_{head} - LL_{tail}$'
    cb.set_label(clbl, fontsize=20.)
    cb.ax.tick_params(labelsize=17)

    # Coast lines

    ax.coastlines(zorder=10)
    ax.set_global()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
        color='black', alpha=0.5, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right=False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black'}# 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black'}# 'weight': 'bold'}
    #gl.xlocator = mticker.FixedLocator([-180., -160, -140, -120, -60, -20.])
    #gl.xlocator = mticker.FixedLocator([-240., -180., -120, -65, -60, -55, 0, 60, 120.])
    #gl.ylocator = mticker.FixedLocator([0., 15., 30., 45, 60.])
    plt.savefig(outfile, dpi = 600)



#### ########################## #########################
def main(pargs):

    # LL histograms
    if pargs.figure == 'LL_histograms':
        fig_LL_histograms(local=pargs.local)

    # Median heads vs tails
    if pargs.figure == 'head_tail':
        fig_med_LL_head_tail()

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("SSL Figures")
    parser.add_argument("figure", type=str, 
                        help="function to execute: 'slopes, 2d_stats, slopevsDT, umap_LL, learning_curve'")
    parser.add_argument('--metric', type=str, help="Metric for the figure: 'DT, T10'")
    parser.add_argument('--cmap', type=str, help="Color map")
    parser.add_argument('--vmnx', default='-1,1', type=str, help="Color bar scale")
    parser.add_argument('--outfile', type=str, help="Outfile")
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

# LL histograms
# python py/figs_llc_viirs.py LL_histograms --local

# Median values heads vs. tails
# python py/figs_llc_viirs.py head_tail