""" Figures for MAE paper """
from dataclasses import replace
from datetime import datetime
import os, sys
import numpy as np
import scipy
from scipy import stats
from urllib.parse import urlparse
import datetime

import argparse

import healpy as hp

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Ellipse


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import h5py

from ulmo import plotting
from ulmo.utils import utils as utils

from ulmo import io as ulmo_io
from ulmo.ssl import single_image as ssl_simage
from ulmo.ssl import ssl_umap
from ulmo.ssl import defs as ssl_defs
from ulmo.utils import image_utils

from IPython import embed

# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import ssl_paper_analy
#sys.path.append(os.path.abspath("../Figures/py"))
#import fig_ssl_modis


def fig_clouds(outfile:str, analy_file:str,
                 local=False, 
                 debug=False, 
                 color='bwr', vmax=None): 
    """ Global geographic plot of the UMAP select range

    Args:
        outfile (str): 
        table (str): 
            Which table to use
        umap_rngs (list): _description_
        local (bool, optional): _description_. Defaults to False.
        nside (int, optional): _description_. Defaults to 64.
        umap_comp (str, optional): _description_. Defaults to 'S0,S1'.
        umap_dim (int, optional): _description_. Defaults to 2.
        debug (bool, optional): _description_. Defaults to False.
        color (str, optional): _description_. Defaults to 'bwr'.
        vmax (_type_, optional): _description_. Defaults to None.
        min_counts (int, optional): Minimum to show in plot.
        show_regions (str, optional): Rectangles for the geographic regions of this 
            Defaults to False.
        absolute (bool, optional):
            If True, show absolute counts instead of relative
    """

    # Load
    data = np.load(analy_file)
    nside = int(data['nside'])

    # Angles
    npix_hp = hp.nside2npix(nside)
    hp_lons, hp_lats = hp.pixelfunc.pix2ang(nside, np.arange(npix_hp), 
                                            lonlat=True)

    rlbl = r"$\log_{10} \; \rm Counts$"
    vmax = None
    color = 'Blues'


   # Figure
    fig = plt.figure(figsize=(12,8))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    CC_plt = [0.05, 0.1, 0.2, 0.5]
    #CC_values = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 
    #             0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for kk, CC in enumerate(CC_plt):
        mt = np.where(np.isclose(data['CC_values'],CC))[0][0]
        #mt = np.where(np.isclose(CC_values,CC))[0][0]
        ax = plt.subplot(gs[kk], projection=tformM)

        hp_events = np.ma.masked_array(data['hp_pix_CC'][:,mt])
        # Mask
        hp_events.mask = [False]*hp_events.size
        bad = hp_events <= 0
        hp_events.mask[bad] = True
        hp_events.data[bad] = 0

        # Proceed
        hp_plot = np.log10(hp_events)

        cm = plt.get_cmap(color)
        # Cut
        good = np.invert(hp_plot.mask)
        img = plt.scatter(x=hp_lons[good],
            y=hp_lats[good],
            c=hp_plot[good], 
            cmap=cm,
            vmin=0.,
            vmax=vmax, 
            s=1,
            transform=tformP)

        # Colorbar
        cb = plt.colorbar(img, orientation='horizontal', pad=0.)
        lbl = rlbl + f'  (CC={CC:.2f})'
        cb.set_label(lbl, fontsize=15.)
        cb.ax.tick_params(labelsize=17)

        # Coast lines
        ax.coastlines(zorder=10)
        ax.add_feature(cartopy.feature.LAND, 
            facecolor='gray', edgecolor='black')
        ax.set_global()

        gl = ax.gridlines(crs=tformP, linewidth=1, 
            color='black', alpha=0.5, linestyle=':', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right=False
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'color': 'black'}# 'weight': 'bold'}
        gl.ylabel_style = {'color': 'black'}# 'weight': 'bold'}

        plotting.set_fontsize(ax, 19.)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_numhp_clouds(outfile:str, analy_file:str):

    # Load
    data = np.load(analy_file)
    nside = int(data['nside'])
    CC_values = data['CC_values']

    N_mins = [10, 30, 100, 300]
    # Figure
    fig = plt.figure(figsize=(12,8))
    plt.clf()

    ax = plt.gca()

    for N_min in N_mins:
        num_hp = []
        for kk in range(CC_values.size):
            gd = np.sum(data['hp_pix_CC'][:,kk] >= N_min)
            num_hp.append(gd)
        # Plot
        ax.plot(CC_values, num_hp, label=f'N_min={N_min}')

    ax.legend(fontsize=15.)
    ax.set_xlabel('Cloud Cover')
    ax.set_ylabel('Number')
    plotting.set_fontsize(ax, 17.)

    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # Uniform, non UMAP gallery for DTall
    if flg_fig & (2 ** 0):
        fig_clouds('fig_clouds.png',
                   '/tank/xavier/Oceanography/Python/ulmo/ulmo/runs/MAE/modis_2020_cloudcover.npz')

    if flg_fig & (2 ** 1):
        fig_numhp_clouds('fig_numhp_clouds.png',
                   '/tank/xavier/Oceanography/Python/ulmo/ulmo/runs/MAE/modis_2020_cloudcover.npz')



# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # Clouds on the sphere
        flg_fig += 2 ** 1  # Number satisfying
        #flg_fig += 2 ** 2  # Gallery of 16 with DT = 4
        #flg_fig += 2 ** 3  # Full set of UMAP galleries
        #flg_fig += 2 ** 4  # Regional + Gallery -- Pacific ECT
        #flg_fig += 2 ** 5  # Regional + Gallery -- Coastal california
        #flg_fig += 2 ** 6  # Matched gallery for DT = 1
        #flg_fig += 2 ** 7  # Matched gallery for boring images
        #flg_fig += 2 ** 8  # Time series
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)