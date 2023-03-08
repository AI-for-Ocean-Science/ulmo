""" Figures for SSL paper on MODIS """
from datetime import datetime
import os, sys
import numpy as np
from urllib.parse import urlparse
import datetime

import argparse
import scipy

import healpy as hp

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.dates as mdates


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy

mpl.rcParams['font.family'] = 'stixgeneral'

import seaborn as sns

import h5py

from ulmo import plotting
from ulmo.utils import utils as utils
from ulmo.utils import table as table_utils


from IPython import embed



def fig_umap_multi_metric(tbl, 
                binx:np.ndarray,
                biny:np.ndarray,
                stat='median', 
                cuts=None,
                percentiles=None,
                table=None,
                local=False, 
                cmap=None,
                vmnx = (-1000., None),
                region=None,
                umap_keys=['US0','US1'],
                umap_dim=2,
                metrics:list=None,
                outfile:str=None,
                debug=False): 
    """ UMAP colored by LL or something else

    Args:
        tbl (pandas.DataFrame): Table to plot
        binx (np.ndarray): x bins of UMAP
        outfile (str, optional): [description]. Defaults to 'fig_umap_LL.png'.
        local (bool, optional): [description]. Defaults to True.
        hist_param (dict, optional): 
            dict describing the histogram to generate and show
        debug (bool, optional): [description]. Defaults to False.

    Raises:
        IOError: [description]
    """
    if outfile is None:
        outfile= f'fig_umap_multi_{stat}.png' 

    num_samples = len(tbl)

    # Histogram
    hist_param = dict(binx=binx, biny=biny)

    # Inputs
    if cmap is None:
        # failed = 'inferno, brg,gnuplot'
        cmap = 'gist_rainbow'
        cmap = 'rainbow'

    if metrics is None:
        metrics = ['DT40', 'stdDT40', 'slope', 'clouds', 'abslat', 'counts']

    # Start the figure
    fig = plt.figure(figsize=(12, 6.5))
    plt.clf()
    gs = gridspec.GridSpec(2, 3)

    a_lbls = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for ss, metric in enumerate(metrics):
        print(f'{ss}: Working on {metric}')
        ax = plt.subplot(gs[ss])
        lmetric, values = table_utils.parse_metric(metric, tbl)
        if 'std' in metric: 
            istat = 'std'
        else:
            istat = stat
        # Do it
        stat2d, xedges, yedges, _ =\
            scipy.stats.binned_statistic_2d(
                tbl[umap_keys[0]], 
                tbl[umap_keys[1]],
                values,
                istat,
                bins=[hist_param['binx'], 
                    hist_param['biny']])
        counts, _, _ = np.histogram2d(
                tbl[umap_keys[0]], 
                tbl[umap_keys[1]],
                bins=[hist_param['binx'], 
                    hist_param['biny']])

        # Require at least 50
        bad_counts = counts < 50
        stat2d[bad_counts] = np.nan
        if 'counts' in metric:
            if metric == 'log10counts':
                counts = np.log10(counts)
            img = ax.pcolormesh(xedges, yedges, 
                             counts.T, cmap=cmap) 
        else:
            img = ax.pcolormesh(xedges, yedges, 
                             stat2d.T, cmap=cmap) 

        # Color bar
        cb = plt.colorbar(img, pad=0., fraction=0.030)
        cb.set_label(lmetric, fontsize=15.)
        #ax.set_xlabel(r'$'+umap_keys[0]+'$')
        #ax.set_ylabel(r'$'+umap_keys[1]+'$')
        ax.set_xlabel(r'$U_0$')
        ax.set_ylabel(r'$U_1$')
        fsz = 14.
        ax.text(0.95, 0.9, a_lbls[ss], transform=ax.transAxes,
              fontsize=14, ha='right', color='k')
        plotting.set_fontsize(ax, fsz)

    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

