""" Figures for SSL paper on MODIS """
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
from ulmo.utils import image_utils

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import ssl_paper_analy
sys.path.append(os.path.abspath("../Figures/py"))
import fig_ssl_modis


def fig_uniform_gallery(outfile:str, table:str, 
                     umap_dim=2,
                     umap_comp='S0,S1',
                     seed=1235,
                     min_pts=1000,
                     in_vmnx=(-2., 2.),
                     nxy=4):

    local=True
    if seed is not None:
        np.random.seed(seed)

    # Load
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, table=table)

    # UMAP
    umap_keys = ssl_paper_analy.gen_umap_keys(
        umap_dim, umap_comp)

    # Outfile


    # Grab the cutouts
    modis_tbl, cutouts, umap_grid = ssl_umap.cutouts_on_umap_grid(
        modis_tbl, nxy, umap_keys, min_pts=min_pts)
    ncutouts = len(cutouts)

    cutouts = [item for item in cutouts if item is not None]

    # Pick 9 at random
    choices = np.random.choice(len(cutouts), size=nxy*nxy, replace=False)

    _, cm = plotting.load_palette()
    fig = plt.figure(figsize=(10,9))
    plt.clf()
    gs = gridspec.GridSpec(nxy,nxy)

    cbar_kws = dict(label=r'$\delta T$ (K)')

    # Color bar
    for ii,choice in enumerate(choices):
        cutout = cutouts[choice]

        # Axis
        ax = plt.subplot(gs[ii])

        if ii == len(choices)-1:
            plt_cbar = True
            ax_cbar = ax.inset_axes([1.1,0.05,0.15,0.8])
        else:
            plt_cbar = False
            ax_cbar = None

        # This is only local
        parsed_s3 = urlparse(cutout.pp_file)
        local_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2',
                                    parsed_s3.path[1:])
        cutout_img = image_utils.grab_image(
            cutout, close=True, local_file=local_file)
                # Limits
        if in_vmnx[0] == -999:
            DT = cutout.T90 - cutout.T10
            vmnx = (-1*DT, DT)
        elif in_vmnx is not None:
            vmnx = in_vmnx
        else:
            imin, imax = cutout_img.min(), cutout_img.max()
            amax = max(np.abs(imin), np.abs(imax))
            vmnx = (-1*amax, amax)

        # Plot
        sns_ax = sns.heatmap(np.flipud(cutout_img), 
                        xticklabels=[], 
                    vmin=vmnx[0], vmax=vmnx[1],
                    yticklabels=[], cmap=cm, 
                    cbar=plt_cbar,
                    cbar_ax=ax_cbar, 
                    cbar_kws=cbar_kws,
                    ax=ax)
        sns_ax.set_aspect('equal', 'datalim')

    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_regional_with_gallery(geo_region:str, outfile:str, table:str, 
                     umap_comp='S0,S1', 
                     umap_dim=2, cmap='bwr', nxy=16):

    # Load
    local=True
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, table=table)

    # UMAP
    umap_keys = ssl_paper_analy.gen_umap_keys(
        umap_dim, umap_comp)


    # UMAP the region
    counts, counts_geo, modis_tbl, grid = ssl_umap.regional_analysis(
        geo_region, modis_tbl, nxy, umap_keys)

    rtio_counts = counts_geo / counts

    # Figure
    _, cm = plotting.load_palette()
    fig = plt.figure(figsize=(10,9))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    ax_regional = plt.subplot(gs[0])

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
        fig_uniform_gallery('fig_uniform_gallery_DTall.png',
            '96clear_v4_DTall', seed=1235)

    # Uniform, non UMAP gallery for DT1
    if flg_fig & (2 ** 1):
        fig_uniform_gallery('fig_uniform_gallery_DT1.png',
                            '96clear_v4_DT1', seed=1236,
                     in_vmnx=(-1, 1.))

    # Uniform, non UMAP gallery for DT4
    if flg_fig & (2 ** 2):
        fig_uniform_gallery('fig_uniform_gallery_DT4.png', 
                            '96clear_v4_DT4', seed=1236, 
                            in_vmnx=(-2, 2.))

    # All the galleries
    if flg_fig & (2 ** 3):
        for vmnx, table, outfile in zip(
            [(-0.5,0.5), 
             (-0.75,0.75),
             (-1.,1.),
             (-1.5,1.5),
             (-2.,2.),
             (-3.,3.),
             ],
            ['96clear_v4_DT0', 
             '96clear_v4_DT1',
             '96clear_v4_DT15',
             '96clear_v4_DT2',
             '96clear_v4_DT4',
             '96clear_v4_DT5',
             ],
            ['fig_umap_gallery_DT0.png',
             'fig_umap_gallery_DT1.png',
             'fig_umap_gallery_DT15.png',
             'fig_umap_gallery_DT2.png',
             'fig_umap_gallery_DT4.png',
             'fig_umap_gallery_DT5.png',
             ]):
            #if 'DT5' not in outfile:
            #    continue
            fig_ssl_modis.fig_umap_gallery(
                in_vmnx=vmnx, table=table, local=True, outfile=outfile,
                umap_dim=2, umap_comp='S0,S1',
                skip_incidence=True, min_pts=200, cut_to_inner=40, nxy=8)

    # Regional, Pacific ECT
    if flg_fig & (2 ** 4):
        fig_regional_with_gallery(
            'eqpacific',
            'fig_regional_with_gallery_eqpacific.png',
            '96clear_v4_DT1')


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # Gallery of 16 with DT = all
        #flg_fig += 2 ** 1  # Gallery of 16 with DT = 1
        #flg_fig += 2 ** 2  # Gallery of 16 with DT = 4
        #flg_fig += 2 ** 3  # Full set of UMAP galleries
        flg_fig += 2 ** 4  # Regional + Gallery -- Pacific ECT
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)