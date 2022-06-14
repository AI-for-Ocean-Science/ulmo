""" Figures for LLC analysis in development"""
import os, sys
from typing import IO
import numpy as np
import glob

from urllib.parse import urlparse

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pandas.core.frame import DataFrame
import xarray

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import healpy as hp

from ulmo import plotting
from ulmo.utils import image_utils
from ulmo.utils import utils as utils
from ulmo.analysis import figures as ulmo_figs
from ulmo.llc import io as llc_io
from ulmo.llc import kinematics

from ulmo import io as ulmo_io

from IPython import embed

local_modis_file = '/home/xavier/Projects/Oceanography/AI/OOD/MODIS_L2/Tables/MODIS_L2_std.parquet'


def fig_xxx_Npos(stat_str:str, log_scale=True): 
    # Load LLC
    LLC_FS_file = 's3://llc/Tables/LLC_FS_r1.0.parquet'
    llc_table = ulmo_io.load_main_table(LLC_FS_file)
    gd = np.isfinite(llc_table.T90.values) 
    llc_table = llc_table[gd].copy()

    # Stat
    if stat_str == 'FS':
        outfile='fig_FS_Npos.png'
        stat = llc_table.FS_Npos.values
        clbl=r'$\log_{10} \, N(F_s > 0.0002)$'
    elif stat_str == 'gradb':
        outfile='fig_gradb_Npos.png'
        stat = llc_table.gradb_Npos.values
        clbl=r'$\log_{10} \, N(|\nabla b|^2 > 0.0002)$'
    else:
        raise IOError(f'Bad stat: {stat_str}')

    # Stats
    if log_scale:
        metric = np.zeros_like(stat)
        gdp = np.where(stat > 0.)
        metric[gdp] = np.log10(stat[gdp]) 
    else:
        metric = stat


    fig = plt.figure(figsize=(12,8))
    plt.clf()

    #tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformP)
    #cm = plt.get_cmap('Blues')
    cm = plt.get_cmap('inferno')
    # Cut
    img = plt.scatter(x=llc_table.lon,
        y=llc_table.lat,
        c=metric,
        cmap=cm,
        s=1/20.,
        transform=tformP)

    cb = plt.colorbar(img, orientation='horizontal', pad=0., location='top')
    cb.set_label(clbl, fontsize=20.)

    # Spatial plot
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
    gl.ylabel_style = {'color': 'black'}#

    #plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f'Wrote: {outfile}')




def fig_LL_distribution(outfile, LL_source='LLC', nside=64, 
                        func='mean', vmnx=None,
                        min_sample=5, normalize=False):
    """Spatial maps of LL metrics

        outfile ([type]): [description]
        nside (int, optional): [description]. Defaults to 64.
        func (str, optional): [description]. Defaults to 'mean'.
            diff_mean -- Subtract <MODIS> - <LLC>
    """
    # Inputs
    if vmnx is None:
        vmnx = (-2000., None)
    # Load Table
    table_file = 's3://llc/Tables/test_noise_modis2012.parquet'
    ulmo_table = ulmo_io.load_main_table(table_file)

    # Do the analysis

    if normalize:
        mean_LL_MODIS = np.mean(ulmo_table.modis_LL.values)
        std_LL_MODIS = np.std(ulmo_table.modis_LL.values)
        mean_LL_LLC = np.mean(ulmo_table.LL.values)
        std_LL_LLC = np.std(ulmo_table.LL.values)
        # Scale the LLC
        Nstd = (ulmo_table.LL.values-mean_LL_LLC)/std_LL_LLC
        ulmo_table['new_LL_LLC'] = mean_LL_MODIS + Nstd*std_LL_MODIS
        # Check
        if False:
            df = pandas.concat(axis=0, ignore_index=True,
                        objs=[
                            pandas.DataFrame.from_dict(dict(LL=ulmo_table.new_LL_LLC,
                                                            Data='New LLC')),
                            pandas.DataFrame.from_dict(dict(LL=ulmo_table.modis_LL,
                                                            Data='MODIS (2012)')),
                        ]
                        )
            ax = sns.histplot(data=df, x='LL', hue='Data')
            embed(header='276 of figs')

    # Grab lats, lons
    lats = ulmo_table.lat.values
    lons = ulmo_table.lon.values

    # Healpix coords
    theta = (90 - lats) * np.pi / 180.
    phi = lons * np.pi / 180.
    hp_idx = hp.pixelfunc.ang2pix(nside, theta, phi)

    # Count events
    npix_hp = hp.nside2npix(nside)

    LL_stat = np.zeros(npix_hp)
    LL_rms = np.zeros(npix_hp)

    uni_idx = np.unique(hp_idx)
    # Looping
    print("Looping...")
    for idx in uni_idx:
        in_hp = idx == hp_idx
        if np.sum(in_hp) < min_sample:
            continue
        # Grab data 
        if normalize:
            LL_LLC = ulmo_table[in_hp].new_LL_LLC.values
        else:
            LL_LLC = ulmo_table[in_hp].LL.values
        LL_MODIS = ulmo_table[in_hp].modis_LL.values
        # Proceed
        if LL_source == 'LLC':
            LL = LL_LLC
        elif LL_source == 'MODIS':
            LL = LL_MODIS
        # Stats
        if func == 'mean':
            LL_stat[idx] = np.mean(LL)
        elif func == 'median':
            LL_stat[idx] = np.median(LL)
        elif func == 'diff_mean':
            LL_stat[idx] = np.mean(LL_MODIS) - np.mean(LL_LLC) 
        else:
            raise IOError("Bad function")
        LL_rms[idx] = np.std(LL)

    # Angles
    hp_lons, hp_lats = hp.pixelfunc.pix2ang(nside, np.arange(npix_hp), lonlat=True)

    # Nan me
    good = LL_stat != 0.

    # Figure time
    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformP)

    cm = plt.get_cmap('seismic')
    #img = ax.tricontourf(hp_lons, hp_lats, LL_stat, 
    #                     transform=tformM,
    #                     levels=20, cmap=cm)#, zorder=10)
    img = plt.scatter(x=hp_lons[good],
            y=hp_lats[good],
            c=LL_stat[good],
            cmap=cm,
            s=1,
            vmin=vmnx[0],
            vmax=vmnx[1],
            transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    clbl=f'{func} LL'
    cb.set_label(clbl, fontsize=20.)
    cb.ax.tick_params(labelsize=17)

    # Coast lines
    ax.coastlines(zorder=10)
    ax.set_global()

    # Layout and save
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))



def set_fontsize(ax,fsz):
    '''
    Generate a Table of columns and so on
    Restrict to those systems where flg_clm > 0

    Parameters
    ----------
    ax : Matplotlib ax class
    fsz : float
      Font size
    '''
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsz)


#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # FS Npos across the globe
    if flg_fig & (2 ** 0):
        fig_xxx_Npos('FS')

    # Front Npos across the globe
    if flg_fig & (2 ** 1):
        fig_xxx_Npos('gradb')

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # FS Npos across the globe
        flg_fig += 2 ** 1  # gradb Npos across the globe
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)

