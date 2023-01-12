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

from ulmo.utils import utils as utils
from ulmo import io as ulmo_io
from ulmo.plotting import plotting 

from ulmo.ssl import single_image as ssl_simage
from ulmo.utils import image_utils
from ulmo.llc import io as llc_io

from IPython import embed

#llc_table = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'

sys.path.append(os.path.abspath("../Analysis/py"))
import sst_compare_utils
import generate_cutouts


def load_hp_files(hp_type, hp_root:str):

    # Load up the data
    if hp_type == 'heads':
        evts_head    = np.load('../Analysis/evts_head'+hp_root, allow_pickle=True)
        meds_head = np.load('../Analysis/meds_head'+hp_root, allow_pickle=True)
        hp_lons_head = np.load('../Analysis/hp_lons_head'+hp_root, allow_pickle=True)
        hp_lats_head = np.load('../Analysis/hp_lats_head'+hp_root, allow_pickle=True)
        return evts_head, meds_head, hp_lons_head, hp_lats_head
    elif hp_type == 'tails':
        evts_tail    = np.load('../Analysis/evts_tail'+hp_root, allow_pickle=True)
        hp_lons_tail = np.load('../Analysis/hp_lons_tail'+hp_root, allow_pickle=True)
        hp_lats_tail = np.load('../Analysis/hp_lats_tail'+hp_root, allow_pickle=True)
        meds_tail = np.load('../Analysis/meds_tail'+hp_root, allow_pickle=True)
        return evts_tail, meds_tail, hp_lons_tail, hp_lats_tail
    elif hp_type == 'all':
        evts    = np.load('../Analysis/evts'+hp_root, allow_pickle=True)
        hp_lons = np.load('../Analysis/hp_lons'+hp_root, allow_pickle=True)
        hp_lats = np.load('../Analysis/hp_lats'+hp_root, allow_pickle=True)
        meds = np.load('../Analysis/meds'+hp_root, allow_pickle=True)
        return evts, meds, hp_lons, hp_lats
    else:
        return None

def fig_LL_histograms(outfile='fig_LL_histograms.png', local=True):
    """ Main histogram figure

    Args:
        outfile (str, optional): _description_. Defaults to 'fig_LL_histograms.png'.
        local (bool, optional): _description_. Defaults to True.
    """

    llc_tbl = sst_compare_utils.load_table('llc_match', local=local)
    viirs_tbl = sst_compare_utils.load_table('viirs', local=local)
    print(f"N VIIRS: {len(viirs_tbl)}")
    print(f"N LLC: {len(llc_tbl)}")

    # Stats
    med_viirs = np.median(viirs_tbl.LL)
    med_LLC = np.nanmedian(llc_tbl.LL)
    print(f"Medians:  VIIRS={med_viirs}, LLC={med_LLC}")

    # Quantiles
    for quantile in [0.25, 0.5, 0.75]:
        q_viirs = np.quantile(viirs_tbl.LL, quantile)
        q_llc = np.quantile(llc_tbl.LL, quantile)
        print(f"Quantiles: q={quantile}, viirs={q_viirs}, llc={q_llc}")


    xmnx = (-1000., 1200.)

    # Cut
    llc_tbl = llc_tbl[llc_tbl.LL > xmnx[0]].copy()
    viirs_tbl = viirs_tbl[viirs_tbl.LL > xmnx[0]].copy()

    # Figure time
    fig = plt.figure(figsize=(7, 4))
    plt.clf()
    ax = plt.gca()

    stat = 'density'
    _ = sns.histplot(llc_tbl, x='LL', ax=ax, label='LLC', stat=stat)
    _ = sns.histplot(viirs_tbl, x='LL', ax=ax, color='orange', 
                     label='VIIRS', stat=stat)

    print(f"VIIRS: {len(viirs_tbl)}")
    print(f"LLC: {len(llc_tbl)}")

    ax.set_xlim(xmnx)

    ax.legend()

    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_explore_highLL(outfile='fig_explore_highLL.png', local=True):
    llc_tbl = sst_compare_utils.load_table('llc_match', local=local)
    llc_tbl['DT'] = llc_tbl.T90.values - llc_tbl.T10.values

    # Cut
    high_LL = llc_tbl.LL > 750.
    high_llc = llc_tbl[high_LL].copy()

    fig = plt.figure(figsize=(3, 5))
    plt.clf()
    gs = gridspec.GridSpec(2,1)

    # First DT 
    axDT = plt.subplot(gs[0])
    _ = sns.histplot(high_llc, x='DT', ax=axDT)

    # Now geographic

    # Healpix me
    hp_events, hp_lons, hp_lats = image_utils.evals_to_healpix(
        high_llc, 64, log=False, mask=True)

        
    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    #ax = plt.axes(projection=tformM)
    axgeo = plt.subplot(gs[1], projection=tformM)

    cm = plt.get_cmap('Reds')
    # Cut
    good = np.invert(hp_events.mask)
    img = axgeo.scatter(x=hp_lons[good],
        y=hp_lats[good],
        c=hp_events[good], vmin=0, vmax=100, 
        cmap=cm,
        s=1,
        transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    clbl = r'$N(\rm LL > 750)$'
    cb.set_label(clbl, fontsize=10.)
    cb.ax.tick_params(labelsize=10)

    # Coast lines

    axgeo.coastlines(zorder=10)
    axgeo.set_global()

    gl = axgeo.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
        color='black', alpha=0.5, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right=False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black'}# 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black'}# 'weight': 'bold'}
    
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_med_LL_head_tail(outfile='med_LL_diff_head_vs_tail.png',
                         hp_root='_v98'):

    evts_head, meds_head, hp_lons_head, hp_lats_head = load_hp_files('heads', hp_root)
    evts_tail, meds_tail, hp_lons_tail, hp_lats_tail = load_hp_files('tails', hp_root)
    
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


def fig_med_LL_VIIRS_LLC(outfile='med_LL_diff_VIIRS_vs_LLC.png',
                         min_counts=10):

    # Load
    evts_v98, meds_v98, hp_lons_v98, hp_lats_v98 = load_hp_files('all', '_v98')
    evts_llc, meds_llc, hp_lons_llc, hp_lats_llc = load_hp_files('all', '_llc_match')

    fig = plt.figure(figsize=(12,8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm = plt.get_cmap('coolwarm')
    # Cut
    good = np.invert(meds_v98.mask) & (evts_v98 >= min_counts)
    img = plt.scatter(x=hp_lons_llc[good],
        y=hp_lats_llc[good],
        c=meds_v98[good]- meds_llc[good], 
        vmin=-300, vmax=300, 
        cmap=cm,
        s=1,
        transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    clbl = r'$LL_{VIIRS} - LL_{LLC}$'
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
    print(f"Wrote: {outfile}")

def fig_viirs_concentration(outfile='viirs_concentration.png'):

    # Load
    evts_v98, meds_v98, hp_lons_v98, hp_lats_v98 = load_hp_files('all', '_v98')
    #evts_llc, meds_llc, hp_lons_llc, hp_lats_llc = load_hp_files('all', '_llc_match')

    fig = plt.figure(figsize=(12,8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm = plt.get_cmap('Reds')
    # Cut
    good = np.invert(evts_v98.mask)
    img = plt.scatter(x=hp_lons_v98[good],
        y=hp_lats_v98[good],
        c=np.log10(evts_v98)[good],
        cmap=cm,
        s=1,
        transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    clbl=r'$\log_{10} \, N_{\rm '+'{}'.format('viirs')+'}$'
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
    plt.savefig(outfile, dpi=300)
    print(f"Wrote: {outfile}")

def fig_viirs_llc_variability(
    outfile='Variability_med_LL_diff_VIIRS_vs_LLC.png'):

    # Load
    evts_v98, meds_v98, hp_lons_v98, hp_lats_v98 = load_hp_files('all', '_v98')
    evts_llc, meds_llc, hp_lons_llc, hp_lats_llc = load_hp_files('all', '_llc_match')

    # Figure
    fig = plt.figure(figsize=(12,8))
    plt.clf()

    # Cut
    a = meds_v98.mask == False
    b = meds_llc.mask == False
    c = a & (a==b)

    plt.scatter(x=evts_v98[c], 
                     y=meds_v98[c]-meds_llc[c], 
                     s=1)

    ax = plt.gca()
    ax.set_xscale('log')

    # Axis Labels
    plt.xlabel('VIIRS-derived Field Count in Bin', fontsize = 20)
    plt.ylabel(r'$LL_{VIIRS} - LL_{LLC}$', fontsize = 20)
    plt.ylim(-1000, 1000)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(outfile, dpi=300)
    print(f"Wrote: {outfile}")

def fig_med_LL_diff_head_vs_tail(
    outfile='med_LL_diff_head_vs_tail.png',
                         min_counts=10):

    # Load
    evts_head, meds_head, hp_lons_head, hp_lats_head = load_hp_files('heads', '_v98')
    evts_tail, meds_tail, hp_lons_tail, hp_lats_tail = load_hp_files('tails', '_v98')

    fig = plt.figure(figsize=(12,8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm = plt.get_cmap('coolwarm')
    # Cut
    good = np.invert(evts_head.mask) & (evts_head > min_counts) & (evts_tail > min_counts)
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
    plt.savefig(outfile, dpi=300) 
    print(f"Wrote: {outfile}")


def fig_gulfstream(outfile='fig_gulfstream.png'):

    # Load
    evts_v98, meds_v98, hp_lons_v98, hp_lats_v98 = load_hp_files('all', '_v98')
    evts_llc, meds_llc, hp_lons_llc, hp_lats_llc = load_hp_files('all', '_llc_match')

    mean = pandas.read_csv('../Analysis/12yrmean.ml'   ,names=['lon','lat'], header=None)
    north= pandas.read_csv('../Analysis/82to86north.ml',names=['lon','lat'], header=None)
    south= pandas.read_csv('../Analysis/82to86south.ml',names=['lon','lat'], header=None)

    fig = plt.figure(figsize=(12,8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm = plt.get_cmap('coolwarm')
    # Cut
    good = np.invert(meds_v98.mask)
    img = plt.scatter(x=hp_lons_llc[good],
        y=hp_lats_llc[good],
        c=meds_v98[good]- meds_llc[good], vmin = -300, vmax = 300, 
        cmap=cm,
        s=500,
        transform=tformP)
    img1 = plt.plot(mean.lon.values,  mean.lat.values,  'k-', transform=tformP)
    img2 = plt.plot(north.lon.values, north.lat.values, 'k-', transform=tformP)
    img3 = plt.plot(south.lon.values, south.lat.values, 'k-', transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.1)
    clbl = r'$LL_{VIIRS} - LL_{LLC}$'
    cb.set_label(clbl, fontsize=20.)
    cb.ax.tick_params(labelsize=17)

    # Coast lines

    ax.coastlines(zorder=10)
    ax.set_extent([-80, -45, 30, 45], ccrs.PlateCarree())
    #ax.set_global()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, x_inline=False, y_inline=False, 
            color='black', alpha=0.5, linestyle=':', draw_labels=True)
    gl.top_labels=False
    gl.bottom_labels=True
    gl.left_labels=True
    gl.right_labels=False
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black', 'size': 14, 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black', 'size': 14, 'weight': 'bold'}

    plt.savefig(outfile, dpi=300)
    print(f"Wrote: {outfile}")

def fig_eq_pacific(outfile='fig_equator_histograms.png',
                   local=False):

    # Load
    if not local:
        v98 = ulmo_io.load_main_table('s3://viirs/Tables/VIIRS_all_98clear_std.parquet')
        llc = ulmo_io.load_main_table('s3://llc/Tables/LLC_uniform144_r0.5.parquet')
    else:
        v98 = ulmo_io.load_main_table(os.path.join(
            os.getenv('SST_OOD'), 'VIIRS', 'Tables', 
            'VIIRS_all_98clear_std.parquet'))
        llc = ulmo_io.load_main_table(os.path.join(
            os.getenv('SST_OOD'), 'LLC', 'Tables', 
            'LLC_uniform144_r0.5.parquet'))

    # DT
    v98['DT'] = v98.T90 - v98.T10
    llc['DT'] = llc.T90 - llc.T10

    # Coords?
    south=0
    north=2
    mid_lon=100
    dlon=5

    # Build it
    rect = (v98.lat > south ) & (v98.lat < north) & (
        np.abs(v98.lon + mid_lon) < dlon)
    viirs_np = v98[ rect ].copy()

    rect = (llc.lat > south ) & (llc.lat < north) & (
        np.abs(llc.lon + mid_lon) < dlon)
    llc_np = llc[ rect ].copy()

    # South
    south=-2
    north=0
    mid_lon=100
    dlon=5

    rect = (v98.lat > south ) & (v98.lat < north) & (
        np.abs(v98.lon + mid_lon) < dlon)
    viirs_sp = v98[ rect ].copy()

    rect = (llc.lat > south ) & (llc.lat < north) & (
        np.abs(llc.lon + mid_lon) < dlon)
    llc_sp = llc[ rect ].copy()

    # Figure
    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(2,2)

    # Above LL
    axI = 0
    axes = []
    sns.set(font_scale = 2)
    for viirs, llc in zip([viirs_np, viirs_sp], [llc_np, llc_sp]):
        for key in ['LL', 'DT']:
            ax = plt.subplot(gs[axI])
            axI += 1
            axes.append(ax)

            sns.histplot(data = viirs, 
                            x=key,
                            binwidth= 20 if key == 'LL' else 0.1, 
                            color='orange', stat='density', 
                            label='VIIRS', ax=ax)
            sns.histplot(data=llc, 
                            x=key,
                            binwidth=20 if key == 'LL' else 0.1, 
                            color='seagreen', 
                            stat='density', label='LLC',
                            ax=ax)

            # Median lines
            if key == 'LL':
                med_viirs = viirs.LL.median()
                med_llc = llc.LL.median()
                ax.axvline(med_viirs, color='orange', 
                            linestyle='--', linewidth=3)
                xoff = 20

                ax.text(med_viirs-xoff, 
                        0.0025, r'$\widetilde{LL}_{\rm VIIRS}$'+f'={int(med_viirs)}',
                        fontsize=15, ha='right', color='orange')
                ax.text(med_llc-xoff, 
                        0.0033, r'$\widetilde{LL}_{\rm LLC}$'+f'={int(med_llc)}',
                        fontsize=15, ha='right', color='seagreen')

                ax.axvline(med_llc, color='seagreen', 
                            linestyle='--', linewidth=3)
            elif key == 'DT':
                ax.set_xlabel(r'$\Delta T$ [K]')
                                
            if axI == 3:
                ax.legend(fontsize=17.)

    fsz = 17.
    for ax in axes:
        plotting.set_fontsize(ax, fsz)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_decile_gallery(local=False, cut=None):

    # Color map
    _, cm = plotting.load_palette()

    # Load
    #v98 = ulmo_io.load_main_table('s3://viirs/Tables/VIIRS_all_98clear_std.parquet')
    #llc = ulmo_io.load_main_table('s3://llc/Tables/llc_viirs_match.parquet')

    llc = sst_compare_utils.load_table('llc_match', local=local)
    v98 = sst_compare_utils.load_table('viirs', local=local)

    # Cut on Temperature?
    T0 = None
    if cut == 'full_5':
        title = '(a) Gallery of the Full Distribution'
        outfile = 'fig_gallery_full_5.png'
        seed = 1236
    elif cut == 'full_10':
        title = '(a) Gallery of the Full Distribution'
        outfile = 'fig_gallery_full_10.png'
        seed = 1237
    elif cut == 'DT125_5':
        title = r'(b) Gallery of $\Delta T = [1,1.5]$ K'
        outfile = 'fig_gallery_DT125_5.png'
        seed = 1236
        T0, dT = 1.25, 0.25
    elif cut == 'DT125_10':
        title = r'(b) Gallery of $\Delta T = [1,1.5]$ K'
        outfile = 'fig_gallery_DT125_10.png'
        seed = 1236
        T0, dT = 1.25, 0.25
    else:
        raise IOError(f"Bad cut: {cut}")

    # T cut?
    if T0 is not None:
        llc = llc[np.abs(llc.DT-T0) < dT]
        v98 = v98[np.abs(v98.DT-T0) < dT]

    # Rnadom seed
    np.random.seed(seed)

    # Indices
    llc.reset_index(drop=True, inplace=True)
    v98.reset_index(drop=True, inplace=True)

    # Figure inputs
    tmin=True
    tmax=True

    # 5 or 10
    # Decile on VIIRS
    # Median LL of decile
    # Random from 50 closest to median

    ndecile = int(cut.split('_')[-1])
    ddecile = 100//ndecile
    pdeciles = np.arange(ddecile, 100+ddecile, ddecile)

    # divvy up cutouts into percentiles
    #l10, l20, l30, l40, l50, l60, l70, l80, l90, l100 = np.percentile(llc.dropna( subset='LL').LL.to_numpy(), [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # v10, v20, v30, v40, v50, v60, v70, v80, v90, v100 = np.percentile(v98.LL.to_numpy(), [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    deciles = np.percentile(v98.LL.to_numpy(), pdeciles)

    '''
    vr60  = v98[ (v98.LL > v50 ) & (v98.LL < v60) ]
    vr70  = v98[ (v98.LL > v60 ) & (v98.LL < v70) ]
    vr80  = v98[ (v98.LL > v70 ) & (v98.LL < v80) ]
    vr90  = v98[ (v98.LL > v80 ) & (v98.LL < v90) ]
    vr100 = v98[ (v98.LL > v90 ) & (v98.LL < v100) ]

    v98_rs = [vr10, vr20, vr30, vr40, vr50, vr60, vr70, vr80, vr90, vr100]
    '''

    # pick 1 cutout from each percentile region
    limgs = []
    vimgs = []

    for kk, decile in enumerate(deciles):
        if kk == 0:
            v_decile  = v98[ (v98.LL.values < decile ) ]
        else:
            v_decile  = v98[ (v98.LL.values < decile ) & (v98.LL.values >= deciles[kk-1]) ]
        # Median
        med_LL = np.median(v_decile.LL.values)

        # Find 50 closest to median
        closest_v = np.abs(v98.LL.values - med_LL).argsort()[:50]
        choice = np.random.choice(closest_v, size = 1)
        vimgs.append(choice[0])

        closest_L = np.abs(llc.LL.values - med_LL).argsort()[:50]
        choice = np.random.choice(closest_L, size = 1)
        limgs.append(choice[0])


    '''
    for reg in llc_rs: 
        img = np.random.choice( reg.index.to_numpy(), size = 1)
        limgs.append(img[0])
        
    for reg in v98_rs:
        img = np.random.choice( reg.index.to_numpy(), size = 1)
        vimgs.append(img[0])
    '''

    # Figure
    ysize = 14 / (10/ndecile)
    fig, axes = plt.subplots(2, ndecile, figsize = (ysize,3) )

    fig.suptitle(title, fontsize=15)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar_kws={"orientation": "vertical", "shrink":1, "aspect":40, "label": "T - T$_{mean}$"}
    pal, cm = plotting.load_palette()

    #determine tmax and tmin
    imgs = np.empty((64,64,20))
    LLs  = np.empty(20)

    for i in range(0,ndecile):
        lidx = limgs[ i ]
        vidx = vimgs[ i ]
        
        lcutout = llc.iloc[ lidx ] 
        vcutout = v98.iloc[ vidx ] 
        
        limg= generate_cutouts.grab_cutout(lcutout, local=local) # llc_io.grab_image(lcutout)
        vimg= generate_cutouts.grab_cutout(vcutout, local=local) # llc_io.grab_image(vcutout)
        
        imgs[:,:,i] = vimg
        imgs[:,:,ndecile + i] = limg
        LLs[i] = vcutout.LL
        LLs[ndecile + i] = lcutout.LL

    if tmax: 
        tmax = np.max(imgs)

    if tmin:
        tmin = np.min(imgs)
    print('Temperature scale is {} to {}.'.format(tmin, tmax))

    # Set by hand?
    tmin, tmax = -2., 2.

    # plot
    for i, ax in enumerate(axes.flat):
        
        # VIIRS
        if i in range(0, ndecile):
            img = imgs[:,:,i]

            sns.heatmap(ax=ax, data=img, xticklabels=[], yticklabels=[], cmap=cm, #'viridis',
                        cbar=i == 0, vmin=tmin, vmax=tmax,
                        cbar_ax=None if i else cbar_ax,
                        cbar_kws=None if i else cbar_kws)

            # Label
            ax.set_title('LL = {}'.format(round(LLs[i])))
            ax.figure.axes[-1].yaxis.label.set_size(15)

        # LLC
        elif i in range(ndecile, ndecile*2):

            img = imgs[:, :, i]
            sns.heatmap(ax=ax, data=img, xticklabels=[], yticklabels=[], cmap=cm, #'viridis',
                        cbar=i == 0, vmin=tmin, vmax=tmax,
                        cbar_ax=None if i else cbar_ax,
                        cbar_kws=None if i else cbar_kws)

        ax.set_aspect('equal', 'datalim')

    fig.tight_layout(rect=[0, 0, .9, 1])

    plt.savefig(outfile, dpi = 300)
    print('Wrote {:s}'.format(outfile))

#### ########################## #########################
def main(pargs):

    # LL histograms
    if pargs.figure == 'LL_histograms':
        fig_LL_histograms(local=pargs.local)

    # Median heads vs tails
    if pargs.figure == 'head_tail':
        fig_med_LL_head_tail()

    # Median LL for VIIRS vs LLC
    if pargs.figure == 'med_LL_VIIRS_LLC':
        fig_med_LL_VIIRS_LLC()

    # Exploring high LL cutouts in LLC
    if pargs.figure == 'explore_highLL':
        fig_explore_highLL()

    # VIIRS geographic location
    if pargs.figure == 'concentration':
        fig_viirs_concentration()

    # VIIRS geographic location
    if pargs.figure == 'variability':
        fig_viirs_llc_variability()

    # Heads/tails
    if pargs.figure == 'med_heads_tails':
        fig_med_LL_diff_head_vs_tail()

    # Gulfstream
    if pargs.figure == 'gulfstream':
        fig_gulfstream()

    # Equatorial Pacific
    if pargs.figure == 'eq_pacific':
        fig_eq_pacific(local=pargs.local)

    # Equatorial Pacific
    if pargs.figure == 'decile_gallery':
        fig_decile_gallery(local=pargs.local,
                           cut=pargs.cut)


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
    parser.add_argument('--cut', type=str, help="Cut ")
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

# Median values VIIRS vs LLC
# python py/figs_llc_viirs.py med_LL_VIIRS_LLC

# Exploring high LL cutouts in LLC
# python py/figs_llc_viirs.py explore_highLL

# VIIRS concentration
# python py/figs_llc_viirs.py concentration

# LLC/VIIRS variability
# python py/figs_llc_viirs.py variability

# VIIRS heads/tails
# python py/figs_llc_viirs.py med_heads_tails

# Gulfstream
# python py/figs_llc_viirs.py gulfstream

# Equatorial Pacific
# python py/figs_llc_viirs.py eq_pacific --local

# Decile gallery
# python py/figs_llc_viirs.py decile_gallery --local