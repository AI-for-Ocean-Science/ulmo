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

# 96
xyrng_dict = {}
#xyrng_dict['xrngs_96_DT15'] = (0., 10.)
#xyrng_dict['yrngs_96_DT15'] = (0., 10.)

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

def grid_umap(U0, U1, nxy=16, percent=[0.1, 99.9]):

    # Boundaries of the grid
    xmin, xmax = np.percentile(U0, percent)
    ymin, ymax = np.percentile(U1, percent)
    dxv = (xmax-xmin)/nxy
    dyv = (ymax-ymin)/nxy

    # Edges
    xmin -= dxv
    xmax += dxv
    ymin -= dyv
    ymax += dyv

    # Grid
    xval = np.arange(xmin, xmax+dxv, dxv)
    yval = np.arange(ymin, ymax+dyv, dyv)

    # Return
    grid = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                xval=xval, yval=yval, dxv=dxv, dyv=dyv)
    return grid


# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import ssl_paper_analy

def update_outfile(outfile, table, umap_dim=2,
                   umap_comp=None):
    # Table
    if table is None or table == 'std':
        pass
    else:
        # Base 1
        if 'CF' in table:
            base1 = '_CF'
        elif '96' in table:
            base1 = '_96clear'
        # DT
        if 'DT' in table:
            dtstr = table.split('_')[1]
            base2 = '_'+dtstr
        else:
            base2 = ''
        outfile = outfile.replace('.png', f'{base1}{base2}.png')
    '''
    elif table == 'CF':
        outfile = outfile.replace('.png', '_CF.png')
    elif table == 'CF_DT0':
        outfile = outfile.replace('.png', '_CF_DT0.png')
    elif table == 'CF_DT1':
        outfile = outfile.replace('.png', '_CF_DT1.png')
    elif table == 'CF_DT15':
        outfile = outfile.replace('.png', '_CF_DT15.png')
    elif table == 'CF_DT1_DT2':
        outfile = outfile.replace('.png', '_CF_DT1_DT2.png')
    elif table == 'CF_DT2':
        outfile = outfile.replace('.png', '_CF_DT2.png')
    '''

    # Ndim
    if umap_dim == 2:
        pass
    elif umap_dim == 3:
        outfile = outfile.replace('.png', '_U3.png')

    # Comps
    if umap_comp is not None:
        if umap_comp != '0,1':
            outfile = outfile.replace('.png', f'_{umap_comp[0]}{umap_comp[-1]}.png')
    # Return
    return outfile
    
def gen_umap_keys(umap_dim, umap_comp):
    if umap_dim == 2:
        if 'T1' in umap_comp:
            umap_keys = ('UT1_'+umap_comp[0], 'UT1_'+umap_comp[-1])
        else:
            ps = umap_comp.split(',')
            umap_keys = ('U'+ps[0], 'U'+ps[-1])
    elif umap_dim == 3:
        umap_keys = ('U3_'+umap_comp[0], 'U3_'+umap_comp[-1])
    return umap_keys


def fig_augmenting(outfile='fig_augmenting.png', use_s3=False):

    # Load up an image
    if use_s3:
        modis_dataset_path = 's3://modis-l2/PreProc/MODIS_R2019_2003_95clear_128x128_preproc_std.h5'
    else:
        modis_dataset_path = os.path.join(os.getenv('SST_OOD'),
                                          "MODIS_L2/PreProc/MODIS_R2019_2003_95clear_128x128_preproc_std.h5")
    with ulmo_io.open(modis_dataset_path, 'rb') as f:
        hf = h5py.File(f, 'r')
        img = hf['valid'][400]

    # Figure time
    _, cm = plotting.load_palette()
    fig = plt.figure(figsize=(7, 2))
    plt.clf()
    gs = gridspec.GridSpec(1,3)

    # No augmentation
    ax0 = plt.subplot(gs[0])
    sns.heatmap(img[0,...], ax=ax0, xticklabels=[], 
                yticklabels=[], cmap=cm, cbar=False)

    # Temperature range
    Trange = img[0,...].min(), img[0,...].max()
    print(f'Temperature range: {Trange}')
    
    # Augment me
    embed(header='187 of figs -- YOU NEED TO NOW UPDATE THE ANGLES IN THE TEXT')
    loader = ssl_simage.image_loader(img)
    test_batch = iter(loader).next()
    img1, img2 = test_batch

    ax1 = plt.subplot(gs[1])
    sns.heatmap(img1[0,0,...], ax=ax1, xticklabels=[], 
                yticklabels=[], cbar=False, cmap=cm,
                vmin=Trange[0], vmax=Trange[1])
    ax2 = plt.subplot(gs[2])
    sns.heatmap(img2[0,0,...], ax=ax2, xticklabels=[], 
                yticklabels=[], cbar=False, cmap=cm,
                vmin=Trange[0], vmax=Trange[1])

    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_umap_colored(outfile='fig_umap_LL.png', 
                cuts=None,
                percentiles=None,
                metric='LL',
                table=None,
                local=False, 
                cmap=None,
                point_size = None, 
                lbl=None,
                vmnx = (-1000., None),
                maxN=None,
                region=None,
                use_std_labels=True,
                hist_param=None,
                umap_comp='0,1',
                umap_dim=2,
                debug=False): 
    """ UMAP colored by LL or something else

    Args:
        outfile (str, optional): [description]. Defaults to 'fig_umap_LL.png'.
        local (bool, optional): [description]. Defaults to True.
        hist_param (dict, optional): 
            dict describing the histogram to generate and show
        debug (bool, optional): [description]. Defaults to False.

    Raises:
        IOError: [description]
    """
    # Load table
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, cuts=cuts, region=region, 
        table=table, percentiles=percentiles)

    # Limit the sample?
    if maxN is not None:
        N = len(modis_tbl)
        idx = np.random.choice(np.arange(N), maxN, replace=False)
        modis_tbl = modis_tbl.iloc[idx].copy()

    num_samples = len(modis_tbl)
    outfile = update_outfile(outfile, table, umap_dim,
                             umap_comp=umap_comp)
    umap_keys = gen_umap_keys(umap_dim, umap_comp)

    # Inputs
    if cmap is None:
        cmap = 'jet'


    if debug: # take a subset
        print("DEBUGGING IS ON")
        nsub = 500000
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        idx = idx[0:nsub]
        modis_tbl = modis_tbl.loc[idx].copy()

    # Metric
    lmetric = metric
    if metric == 'LL':
        values = modis_tbl.LL 
    elif metric == 'logDT':
        values = np.log10(modis_tbl.DT.values)
        lmetric = r'$\log \, \Delta T$'
    elif metric == 'DT':
        values = modis_tbl.DT.values
        lmetric = r'$\Delta T$'
    elif metric == 'clouds':
        values = modis_tbl.clear_fraction
    elif metric == 'slope':
        values = modis_tbl.min_slope.values
    else:
        raise IOError("Bad metric!")
    
    # Histogram??
    if hist_param is not None:
        stat, xedges, yedges, _ =\
            stats.binned_statistic_2d(
                modis_tbl[umap_keys[0]], 
                modis_tbl[umap_keys[1]],
                values,
                'median', # 'std', 
                bins=[hist_param['binx'], 
                    hist_param['biny']])
        counts, _, _ = np.histogram2d(
                modis_tbl[umap_keys[0]], 
                modis_tbl[umap_keys[1]],
                bins=[hist_param['binx'], 
                    hist_param['biny']])


    # Start the figure
    fig = plt.figure(figsize=(8, 8))
    plt.clf()
    gs = gridspec.GridSpec(1, 1)

    # Just the UMAP colored by one of the stats
    ax0 = plt.subplot(gs[0])

    if point_size is None:
        point_size = 1. / np.sqrt(num_samples)
    if hist_param is None:
        img = ax0.scatter(modis_tbl[umap_keys[0]], modis_tbl[umap_keys[1]],
            s=point_size, c=values,
            cmap=cmap, vmin=vmnx[0], vmax=vmnx[1])
    else:
        bad_counts = counts < 50
        stat[bad_counts] = np.nan
        img = ax0.pcolormesh(xedges, yedges, 
                             stat.T, cmap=cmap) 

    # Color bar
    cb = plt.colorbar(img, pad=0., fraction=0.030)
    cb.set_label(lmetric, fontsize=14.)
    #
    if use_std_labels:
        ax0.set_xlabel(r'$U_0$')
        ax0.set_ylabel(r'$U_1$')
    else:
        ax0.set_xlabel(r'$'+umap_keys[0]+'$')
        ax0.set_ylabel(r'$'+umap_keys[1]+'$')
    #ax0.set_aspect('equal')#, 'datalim')

    fsz = 17.
    plotting.set_fontsize(ax0, fsz)

    # Label
    if lbl is not None:
        ax0.text(0.05, 0.9, lbl, transform=ax0.transAxes,
              fontsize=15, ha='left', color='k')

    #plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_umap_density(outfile='fig_umap_density.png',
                     local=False, table='std', 
                     umap_comp='0,1', umap_grid=None,
                     umap_dim=2, cmap=None, nxy=16,
                     fsz=19.,
                     modis_tbl=None,
                     use_std_lbls=True,
                     show_cbar = False,
                     debug=False, ax=None): 
    # Load
    if modis_tbl is None:
        modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, table=table)

    umap_keys = gen_umap_keys(umap_dim, umap_comp)
    if outfile is not None:
        outfile = update_outfile(outfile, table, umap_dim,
                             umap_comp=umap_comp)

    # Boundaries of the box
    if umap_grid is None:
        grid_umap(modis_tbl[umap_keys[0]].values, modis_tbl[umap_keys[0]].values,
                  nxy=nxy)

    xmin, xmax = umap_grid['xmin'], umap_grid['xmax']
    ymin, ymax = umap_grid['ymin'], umap_grid['ymax']
    dxv = umap_grid['dxv']
    dyv = umap_grid['dyv']

    # Grid
    xval = np.arange(xmin, xmax+dxv, dxv)
    yval = np.arange(ymin, ymax+dyv, dyv)

    # cut
    good = (modis_tbl[umap_keys[0]] > xmin) & (
        modis_tbl[umap_keys[0]] < xmax) & (
        modis_tbl[umap_keys[1]] > ymin) & (
            modis_tbl[umap_keys[1]] < ymax) & np.isfinite(modis_tbl.LL)

    modis_tbl = modis_tbl.loc[good].copy()
    num_samples = len(modis_tbl)
    print(f"We have {num_samples} making the cuts.")

    counts, xedges, yedges = np.histogram2d(
        modis_tbl[umap_keys[0]], 
        modis_tbl[umap_keys[1]], bins=(xval, yval))

    counts /= np.sum(counts)

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        plt.clf()
        ax = plt.gca()


    if use_std_lbls:
        ax.set_xlabel(r'$U_0$')
        ax.set_ylabel(r'$U_1$')
    else:
        ax.set_xlabel(r'$'+umap_keys[0]+'$')
        ax.set_ylabel(r'$'+umap_keys[1]+'$')

    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(ymin, ymax)

    if cmap is None:
        cmap = "Greys"
    cm = plt.get_cmap(cmap)
    values = counts.transpose()
    lbl = 'Counts'
    vmax = None
    mplt = ax.pcolormesh(xedges, yedges, values, 
                         cmap=cm, 
                         vmax=vmax) 

    # Color bar
    if show_cbar:
        cbaxes = plt.colorbar(mplt, pad=0., fraction=0.030)
        #cbaxes.set_label(lbl, fontsize=15.)

    plotting.set_fontsize(ax, fsz)

    # Write?
    if outfile is not None:
        plt.savefig(outfile, dpi=200)
        plt.close()
        print('Wrote {:s}'.format(outfile))
    

def fig_umap_gallery(outfile='fig_umap_gallery_vmnx5.png',
                     local=False, table='std', in_vmnx=None,
                     umap_comp='0,1', nxy=16,
                     min_pts=10,
                     umap_dim=2,
                     use_std_lbls=True,
                     debug=False): 
    """ UMAP gallery

    Args:
        outfile (str, optional): [description]. Defaults to 'fig_umap_LL.png'.
        version (int, optional): [description]. Defaults to 1.
        local (bool, optional): [description]. Defaults to True.
        debug (bool, optional): [description]. Defaults to False.

    Raises:
        IOError: [description]
    """
    # Load
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, table=table)

    umap_keys = gen_umap_keys(umap_dim, umap_comp)
    outfile = update_outfile(outfile, table, umap_dim,
                             umap_comp=umap_comp)

    if debug:
        nxy = 4

    # Cut table
    dxv = 0.5
    dyv = 0.25
    if table == 'CF' and umap_dim==2:
        xmin, xmax = xrngs_CF
        ymin, ymax = yrngs_CF
    elif table == 'CF_DT2' and umap_dim==2:
        xmin, xmax = xrngs_CF_DT2
        ymin, ymax = yrngs_CF_DT2
    elif table == 'CF_DT0' and umap_dim==2:
        xmin, xmax = xrngs_CF_DT0
        ymin, ymax = yrngs_CF_DT0
        dyv = 0.5
    elif table == 'CF_DT1' and umap_dim==2:
        xmin, xmax = xrngs_CF_DT1
        ymin, ymax = yrngs_CF_DT1
        dxv = 0.25
    elif table == 'CF_DT15' and umap_dim==2:
        xmin, xmax = xrngs_CF_DT15
        ymin, ymax = yrngs_CF_DT15
        dxv = 0.5 * 0.8
        dyv = dxv * 8./10
    elif table == 'CF_DT1_DT2' and umap_dim==2:
        xmin, xmax = xrngs_CF_DT1
        ymin, ymax = yrngs_CF_DT1
        dxv = 0.5
        dyv = 0.5
    elif table == 'CF' and umap_dim==3 and umap_comp=='0,1':
        xmin, xmax = xrngs_CF_U3
        ymin, ymax = yrngs_CF_U3
    elif table == 'CF' and umap_dim==3 and umap_comp=='1,2':
        xmin, xmax = xrngs_CF_U3_12
        ymin, ymax = yrngs_CF_U3_12
        dxv = 0.25
        # Add more!
        dyv *= 0.66
        dxv *= 0.66
    elif '96_DT' in table: 
        if f'xrngs_{table}' in xyrng_dict.keys():
            xmin, xmax = xyrng_dict[f'xrngs_{table}']
            ymin, ymax = xyrng_dict[f'yrngs_{table}']
            dxv = 0.5 
            dyv = 0.25
        else:
            umap_grid = grid_umap(modis_tbl[umap_keys[0]].values,
                                  modis_tbl[umap_keys[1]].values, nxy=nxy)
            # Unpack
            xmin, xmax = umap_grid['xmin'], umap_grid['xmax']
            ymin, ymax = umap_grid['ymin'], umap_grid['ymax']
            dxv = umap_grid['dxv']
            dyv = umap_grid['dyv']
    else:
        xmin, xmax = -4.5, 7
        ymin, ymax = 4.5, 10.5

    # cut
    good = (modis_tbl[umap_keys[0]] > xmin) & (
        modis_tbl[umap_keys[0]] < xmax) & (
        modis_tbl[umap_keys[1]] > ymin) & (
            modis_tbl[umap_keys[1]] < ymax) & np.isfinite(modis_tbl.LL)

    # Hack for now
    if table == 'CF_DT1_DT2':
        gd = (modis_tbl.UT1_0 != 0.) & (modis_tbl.T90-modis_tbl.T10 > 2.)
        good = good & gd

    modis_tbl = modis_tbl.loc[good].copy()
    num_samples = len(modis_tbl)
    print(f"We have {num_samples} making the cuts.")

    if debug: # take a subset
        print("DEBUGGING IS ON")
        nsub = 500000
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        idx = idx[0:nsub]
        modis_tbl = modis_tbl.iloc[idx].copy()

    # Fig
    _, cm = plotting.load_palette()
    fsz = 15.
    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    ax_gallery = fig.add_axes([0.05, 0.1, 0.6, 0.90])

    if use_std_lbls:
        ax_gallery.set_xlabel(r'$U_0$')
        ax_gallery.set_ylabel(r'$U_1$')
    else:
        ax_gallery.set_xlabel(r'$'+umap_keys[0]+'$')
        ax_gallery.set_ylabel(r'$'+umap_keys[1]+'$')

    # Gallery
    #dxdy=(0.3, 0.3)
    #xmin, xmax = modis_tbl.U0.min()-dxdy[0], modis_tbl.U0.max()+dxdy[0]
    #ymin, ymax = modis_tbl.U1.min()-dxdy[1], modis_tbl.U1.max()+dxdy[1]
    ax_gallery.set_xlim(xmin, xmax)
    ax_gallery.set_ylim(ymin, ymax)

    print('x,y', xmin, xmax, ymin, ymax, dxv, dyv)

    
    # ###################
    # Gallery time

    # Grid
    xval = np.arange(xmin, xmax+dxv, dxv)
    yval = np.arange(ymin, ymax+dyv, dyv)

    # Ugly for loop
    ndone = 0
    if debug:
        nmax = 100
    else:
        nmax = 1000000000

    # Color bar
    plt_cbar = True
    ax_cbar = ax_gallery.inset_axes(
                    [xmax + dxv/10, ymin, dxv/2, (ymax-ymin)*0.2],
                    transform=ax_gallery.transData)
    cbar_kws = dict(label=r'$\Delta T$ (K)')

    for x in xval[:-1]:
        for y in yval[:-1]:
            pts = np.where((modis_tbl[umap_keys[0]] >= x) & (
                modis_tbl[umap_keys[0]] < x+dxv) & (
                modis_tbl[umap_keys[1]] >= y) & (modis_tbl[umap_keys[1]] < y+dxv)
                           & np.isfinite(modis_tbl.LL))[0]
            if len(pts) < min_pts:
                continue

            # Pick a random one
            ichoice = np.random.choice(len(pts), size=1)
            idx = int(pts[ichoice])
            cutout = modis_tbl.iloc[idx]

            # Image
            axins = ax_gallery.inset_axes(
                    [x, y, 0.9*dxv, 0.9*dyv], 
                    transform=ax_gallery.transData)
            try:
                if local:
                    parsed_s3 = urlparse(cutout.pp_file)
                    local_file = os.path.join(os.getenv('SST_OOD'),
                                              'MODIS_L2',
                                              parsed_s3.path[1:])
                    cutout_img = image_utils.grab_image(
                        cutout, close=True, local_file=local_file)
                else:
                    cutout_img = image_utils.grab_image(cutout, close=True)
            except:
                embed(header='198 of plotting')                                                    
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
                     yticklabels=[], cmap=cm, cbar=plt_cbar,
                     cbar_ax=ax_cbar, cbar_kws=cbar_kws,
                     ax=axins)
            # Only do this once
            if plt_cbar:
                plt_cbar = False
            ndone += 1
            print(f'ndone= {ndone}, LL={cutout.LL}')
            if ndone > nmax:
                break
        if ndone > nmax:
            break

    plotting.set_fontsize(ax_gallery, fsz)
    #ax.set_aspect('equal', 'datalim')
    #ax.set_aspect('equal')#, 'datalim')

    # Incidence plot
    ax_incidence = fig.add_axes([0.71, 0.45, 0.25, 0.36])

    fig_umap_density(outfile=None, modis_tbl=modis_tbl,
                     umap_grid=umap_grid, umap_comp=umap_comp,
                     show_cbar=True, ax=ax_incidence, fsz=12.)
    #ax_incidence.plot(np.arange(10), np.arange(10))

    #plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_umap_2dhist(outfile='fig_umap_2dhist.png',
                    table=None,
                    version=1, local=False, vmax=None, 
                    cmap=None, cuts=None, region=None,
                    scl = 1):
    """ Show a 2d histogram of the counts in each cell`

    Args:
        outfile (str, optional): _description_. Defaults to 'fig_umap_2dhist.png'.
        table (_type_, optional): _description_. Defaults to None.
        version (int, optional): _description_. Defaults to 1.
        local (bool, optional): _description_. Defaults to False.
        vmax (_type_, optional): _description_. Defaults to None.
        cmap (_type_, optional): _description_. Defaults to None.
        cuts (_type_, optional): _description_. Defaults to None.
        region (_type_, optional): _description_. Defaults to None.
        scl (int, optional): _description_. Defaults to 1.
    """

    # Load
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, cuts=cuts, table=table,
                               region=region)

    # 
    if pargs.table == 'CF':
        xmin, xmax = xrngs_CF
        ymin, ymax = yrngs_CF
    else:
        xmin, xmax = xrngs_95
        ymin, ymax = yrngs_95
    # Histogram
    bins_U0 = np.linspace(xmin, xmax, 23*scl)
    bins_U1 = np.linspace(ymin,ymax, 24*scl)
    counts, xedges, yedges = np.histogram2d(modis_tbl.U0, modis_tbl.U1,
                                            bins=(bins_U0, bins_U1))

    fig = plt.figure(figsize=(12, 12))
    plt.clf()
    ax = plt.gca()

    if cmap is None:
        cmap = "Greys"
    cm = plt.get_cmap(cmap)
    values = counts.transpose()
    lbl = 'Counts'
    mplt = ax.pcolormesh(xedges, yedges, values, 
                         cmap=cm, 
                         vmax=vmax) 

    # Color bar
    #cbaxes = fig.add_axes([0.03, 0.1, 0.05, 0.7])
    cbaxes = plt.colorbar(mplt, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=15.)
    #cb.set_label(lbl, fontsize=20.)
    #cbaxes.yaxis.set_ticks_position('left')

    ax.set_xlabel(r'$U_0$')
    ax.set_ylabel(r'$U_1$')

    plotting.set_fontsize(ax, 19.)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_umap_geo(outfile, table, umap_rngs, local=False, 
    nside=64, umap_comp='S0,S1', umap_dim=2, debug=False,
    color='bwr', vmax=None): 

    # Load
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, table=table)

    umap_keys = gen_umap_keys(umap_dim, umap_comp)
    outfile = update_outfile(outfile, table, umap_dim,
                             umap_comp=umap_comp)
                            
    # Evaluate full table in healpix
    hp_events, hp_lons, hp_lats = image_utils.evals_to_healpix(modis_tbl, nside)

    # Now the cut region
    cut = ( (modis_tbl[umap_keys[0]] > umap_rngs[0][0]) & 
            (modis_tbl[umap_keys[0]] < umap_rngs[0][1]) & 
            (modis_tbl[umap_keys[1]] > umap_rngs[1][0]) & 
            (modis_tbl[umap_keys[1]] < umap_rngs[1][1]) )
    cut_tbl = modis_tbl[cut].copy()

    print(f"We have {len(cut_tbl)} cutouts in the UMAP range.")

    hp_events_cut, _, _ = image_utils.evals_to_healpix(cut_tbl, nside)

    # Have 0 for unmasked in full set
    masked_in_cut_only = hp_events_cut.mask & np.invert(hp_events.mask)
    hp_events_cut.mask[masked_in_cut_only] = False
    hp_events_cut.data[masked_in_cut_only] = 0.

    # Stats
    f_tot = hp_events / np.sum(hp_events)
    f_cut = hp_events_cut / np.sum(hp_events_cut)

    #embed(header='638 of figs')

    # Ratio
    ratio = f_cut / f_tot #hp_events_cut / hp_events

    # Set 1 event to ratio of 1
    #set_one = (hp_events_cut <= 2) & (hp_events < 10)
    #ratio[set_one] = 1.

    # What to plot?
    hp_plot = ratio
    vmax = 2.


   # Figure
    fig = plt.figure(figsize=(12,8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm = plt.get_cmap(color)
    # Cut
    good = np.invert(hp_plot.mask)
    img = plt.scatter(x=hp_lons[good],
        y=hp_lats[good],
        c=hp_plot[good], 
        cmap=cm,
        vmax=vmax, 
        s=1,
        transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    lbl = "Relative Frequency"
    if lbl is not None:
        cb.set_label(lbl, fontsize=20.)
    cb.ax.tick_params(labelsize=17)

    # Coast lines
    ax.coastlines(zorder=10)
    ax.add_feature(cartopy.feature.LAND, 
        facecolor='gray', edgecolor='black')
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

    plotting.set_fontsize(ax, 19.)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_geo_umap(outfile, geo_region,
                     local=False, 
                     umap_comp='S0,S1',
                     table='96_DT15',
                     umap_dim=2, cmap='bwr',
                     debug=False): 
    # Load
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, table=table)

    umap_keys = gen_umap_keys(umap_dim, umap_comp)
    outfile = update_outfile(outfile, table, umap_dim,
                             umap_comp=umap_comp)
    # Grid
    grid = grid_umap(modis_tbl[umap_keys[0]].values, 
        modis_tbl[umap_keys[1]].values)
 
    # cut
    good = (modis_tbl[umap_keys[0]] > grid['xmin']) & (
        modis_tbl[umap_keys[0]] < grid['xmax']) & (
        modis_tbl[umap_keys[1]] > grid['ymin']) & (
            modis_tbl[umap_keys[1]] < grid['ymax']) & np.isfinite(modis_tbl.LL)

    modis_tbl = modis_tbl.loc[good].copy()
    num_samples = len(modis_tbl)
    print(f"We have {num_samples} making the cuts.")

    # All
    counts, xedges, yedges = np.histogram2d(
        modis_tbl[umap_keys[0]], 
        modis_tbl[umap_keys[1]], bins=(grid['xval'], 
                                       grid['yval']))

    # Normalize
    counts /= np.sum(counts)

    # Geographic
    lons = ssl_paper_analy.geo_regions[geo_region]['lons']
    lats = ssl_paper_analy.geo_regions[geo_region]['lats']
    #embed(header='739 of figs')
    geo = ( (modis_tbl.lon > lons[0]) &
        (modis_tbl.lon < lons[1]) &
        (modis_tbl.lat > lats[0]) &
        (modis_tbl.lat < lats[1]) )

    geo_tbl = modis_tbl.loc[good & geo].copy()
    counts_geo, xedges, yedges = np.histogram2d(
        geo_tbl[umap_keys[0]], 
        geo_tbl[umap_keys[1]], bins=(grid['xval'], 
                                     grid['yval']))
    # Normalize
    counts_geo /= np.sum(counts_geo)

    # Ratio
    rtio_counts = counts_geo / counts

    # Plot
    fig = plt.figure(figsize=(8, 8))
    plt.clf()
    ax = plt.gca()


    ax.set_xlabel(r'$'+umap_keys[0]+'$')
    ax.set_ylabel(r'$'+umap_keys[1]+'$')

    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(ymin, ymax)

    cm = plt.get_cmap(cmap)
    values = rtio_counts.transpose()
    lbl = 'Relative Frequency'
    vmin, vmax = 0, 2.
    mplt = ax.pcolormesh(xedges, yedges, values, 
                         cmap=cm, vmin=vmin, vmax=vmax) 

    # Color bar
    show_cbar = False
    if show_cbar:
        cbaxes = plt.colorbar(mplt, pad=0., fraction=0.030)
        cbaxes.set_label(lbl, fontsize=15.)

    plotting.set_fontsize(ax, 19.)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print('Wrote {:s}'.format(outfile))
    
def fig_seasonal_geo_umap(outfile, geo_region,
                     local=False, 
                     rtio_cut = 1.5,
                     umap_comp='S0,S1',
                     table='96_DT15',
                     umap_dim=2, cmap='bwr',
                     debug=False): 
    # Load
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, table=table)

    umap_keys = gen_umap_keys(umap_dim, umap_comp)
    outfile = update_outfile(outfile, table, umap_dim,
                             umap_comp=umap_comp)
    # Grid
    grid = grid_umap(modis_tbl[umap_keys[0]].values, 
        modis_tbl[umap_keys[1]].values)
 
    # cut
    good = (modis_tbl[umap_keys[0]] > grid['xmin']) & (
        modis_tbl[umap_keys[0]] < grid['xmax']) & (
        modis_tbl[umap_keys[1]] > grid['ymin']) & (
            modis_tbl[umap_keys[1]] < grid['ymax']) & np.isfinite(modis_tbl.LL)

    modis_tbl = modis_tbl.loc[good].copy()
    num_samples = len(modis_tbl)
    print(f"We have {num_samples} making the cuts.")

    # All
    counts, xedges, yedges = np.histogram2d(
        modis_tbl[umap_keys[0]], 
        modis_tbl[umap_keys[1]], bins=(grid['xval'], 
                                       grid['yval']))

    # Normalize
    counts /= np.sum(counts)

    # Geographic
    lons = ssl_paper_analy.geo_regions[geo_region]['lons']
    lats = ssl_paper_analy.geo_regions[geo_region]['lats']
    geo = ( (modis_tbl.lon > lons[0]) &
        (modis_tbl.lon < lons[1]) &
        (modis_tbl.lat > lats[0]) &
        (modis_tbl.lat < lats[1]) )

    geo_tbl = modis_tbl.loc[good & geo].copy()
    counts_geo, xedges, yedges = np.histogram2d(
        geo_tbl[umap_keys[0]], 
        geo_tbl[umap_keys[1]], bins=(grid['xval'], 
                                     grid['yval']))
    # Normalize
    counts_geo /= np.sum(counts_geo)

    # Ratio
    rtio_counts = counts_geo / counts

    if rtio_cut >= 1.:
        use_grid = rtio_counts > rtio_cut
    else:
        embed(header='858 of figs')

    # Loop on years
    months = 1 + np.arange(12)

    pdates = pandas.DatetimeIndex(geo_tbl.datetime)
    fracs = []
    for month in months:
        in_month = pdates.month == month
        month_tbl = geo_tbl[in_month].copy()
        counts_month, xedges, yedges = np.histogram2d(
            month_tbl[umap_keys[0]], 
            month_tbl[umap_keys[1]], 
            bins=(grid['xval'], grid['yval']))
        # frac
        frac = np.sum(counts_month*use_grid) / np.sum(counts_month)
        fracs.append(frac)

    # Plot
    fig = plt.figure(figsize=(8, 8))
    plt.clf()
    ax = plt.gca()

    ax.plot(months, fracs, 'b')

    # Label
    ax.set_ylabel('Fraction')
    ax.set_xlabel('Month')

    plotting.set_fontsize(ax, 19.)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print('Wrote {:s}'.format(outfile))
    


def fig_yearly_geo_umap(outfile, geo_region,
                     local=False, 
                     rtio_cut = 1.5,
                     rtio_region=None,
                     umap_comp='S0,S1',
                     table='96_DT15',
                     umap_dim=2, cmap='bwr',
                     debug=False): 
    # Init
    if rtio_region is None:
        rtio_region = geo_region
    # Load
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, table=table)

    umap_keys = gen_umap_keys(umap_dim, umap_comp)
    outfile = update_outfile(outfile, table, umap_dim,
                             umap_comp=umap_comp)
    # Grid
    grid = grid_umap(modis_tbl[umap_keys[0]].values, 
        modis_tbl[umap_keys[1]].values)
 
    # cut on UMAP space
    good = (modis_tbl[umap_keys[0]] > grid['xmin']) & (
        modis_tbl[umap_keys[0]] < grid['xmax']) & (
        modis_tbl[umap_keys[1]] > grid['ymin']) & (
            modis_tbl[umap_keys[1]] < grid['ymax']) & np.isfinite(modis_tbl.LL)

    modis_tbl = modis_tbl.loc[good].copy()
    num_samples = len(modis_tbl)
    print(f"We have {num_samples} making the UMAP cuts.")

    # All
    counts, xedges, yedges = np.histogram2d(
        modis_tbl[umap_keys[0]], 
        modis_tbl[umap_keys[1]], bins=(grid['xval'], 
                                       grid['yval']))

    # Normalize
    counts /= np.sum(counts)

    # Ratio table

    # Cut on Geography
    lons = ssl_paper_analy.geo_regions[rtio_region]['lons']
    lats = ssl_paper_analy.geo_regions[rtio_region]['lats']
    rtio_geo = ( (modis_tbl.lon > lons[0]) &
        (modis_tbl.lon < lons[1]) &
        (modis_tbl.lat > lats[0]) &
        (modis_tbl.lat < lats[1]) )
    rtio_tbl = modis_tbl.loc[good & rtio_geo].copy()
    
    counts_rtio, xedges, yedges = np.histogram2d(
        rtio_tbl[umap_keys[0]], 
        rtio_tbl[umap_keys[1]], bins=(grid['xval'], 
                                     grid['yval']))
    # Normalize
    counts_rtio /= np.sum(counts_rtio)

    # Ratio
    rtio_counts = counts_rtio / counts

    if rtio_cut >= 1.:
        use_grid = rtio_counts > rtio_cut
    else:
        embed(header='858 of figs')

    # Geo table
    lons = ssl_paper_analy.geo_regions[geo_region]['lons']
    lats = ssl_paper_analy.geo_regions[geo_region]['lats']
    geo = ( (modis_tbl.lon > lons[0]) &
        (modis_tbl.lon < lons[1]) &
        (modis_tbl.lat > lats[0]) &
        (modis_tbl.lat < lats[1]) )
    geo_tbl = modis_tbl.loc[good & geo].copy()

    # Time-series
    years = 2003 + np.arange(17)
    months = 1 + np.arange(12)

    # Loop over each month
    fracs = []
    dates = []
    for year in years:
        for month in months:
            if month < 12:
                in_date = (geo_tbl.datetime >= datetime.datetime(year,month,1)) & (
                    geo_tbl.datetime < datetime.datetime(year,month+1,1))
            else:
                in_date = (geo_tbl.datetime >= datetime.datetime(year,month,1)) & (
                    geo_tbl.datetime < datetime.datetime(year+1,1,1))
            date_tbl = geo_tbl[in_date].copy()
            counts_date, xedges, yedges = np.histogram2d(
                date_tbl[umap_keys[0]], 
                date_tbl[umap_keys[1]], 
                bins=(grid['xval'], grid['yval']))
            # frac
            frac = np.sum(counts_date*use_grid) / np.sum(counts_date)
            fracs.append(frac)
            # Date
            dates.append(datetime.datetime(year, month, 15))

    # Annual
    year_fracs = []
    year_dates = []
    for year in years:
        in_year = (geo_tbl.datetime >= datetime.datetime(year,1,1)) & (
            geo_tbl.datetime < datetime.datetime(year+1,1,1))
        year_tbl = geo_tbl[in_year].copy()
        counts_year, xedges, yedges = np.histogram2d(
            year_tbl[umap_keys[0]], 
            year_tbl[umap_keys[1]], 
            bins=(grid['xval'], grid['yval']))
        # frac
        frac = np.sum(counts_year*use_grid) / np.sum(counts_year)
        year_fracs.append(frac)
        #
        year_dates.append(datetime.datetime(year, 7, 1))

    # Plot
    fig = plt.figure(figsize=(8, 8))
    plt.clf()
    ax = plt.gca()

    # All
    ax.plot(dates, fracs, 'k')

    # Annual
    ax.plot(year_dates, year_fracs, 'ro')

    plotting.set_fontsize(ax, 19.)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print('Wrote {:s}'.format(outfile))
    

def fig_LLvsDT(outfile='fig_LLvsDT.png', local=False, vmax=None, 
                umap_dim=2,
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
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, cuts=cuts,
                                               table=table)
    outfile = update_outfile(outfile, table, umap_dim)

    # Debug?
    if debug:
        modis_tbl = modis_tbl.loc[np.arange(1000000)].copy()

    # Plot
    fig = plt.figure()#figsize=(9, 12))
    plt.clf()

    ymnx = [-5000., 1000.]

    jg = sns.jointplot(data=modis_tbl, x='DT', y='LL', kind='hex',
                       bins='log', gridsize=250, xscale='log',
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
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_slopevsDT(outfile='fig_slopevsDT.png', table=None,
                  local=False, vmax=None, 
                    cmap=None, cuts=None, scl = 1, debug=False):
    """ Bivariate of slope_min vs. DT

    Args:
        outfile (str, optional): [description]. Defaults to 'fig_slopevsDT.png'.
        local (bool, optional): [description]. Defaults to False.
        vmax ([type], optional): [description]. Defaults to None.
        cmap ([type], optional): [description]. Defaults to None.
        cuts ([type], optional): [description]. Defaults to None.
        scl (int, optional): [description]. Defaults to 1.
        debug (bool, optional): [description]. Defaults to False.
    """

    # Load table
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, cuts=cuts)
    outfile = update_outfile(outfile, table)

    # Debug?
    if debug:
        modis_tbl = modis_tbl.loc[np.arange(1000000)].copy()

    # Plot
    fig = plt.figure(figsize=(12, 12))
    plt.clf()


    jg = sns.jointplot(data=modis_tbl, x='DT', y='min_slope', kind='hex',
                       bins='log', gridsize=250, xscale='log',
                       cmap=plt.get_cmap('winter'), mincnt=1,
                       marginal_kws=dict(fill=False, color='black', bins=100)) 
    jg.ax_joint.set_xlabel(r'$\Delta T$')
    jg.ax_joint.set_ylabel(metric_lbls['min_slope'])

    plotting.set_fontsize(jg.ax_joint, 15.)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_slopes(outfile='fig_slopes.png', 
               local=False, vmax=None, table=None,
                    cmap=None, cuts=None, scl = 1, debug=False):

    # Load table
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, cuts=cuts)
    outfile = update_outfile(outfile, table)

    # Debug?
    if debug:
        modis_tbl = modis_tbl.loc[np.arange(100000)].copy()

    # Check on isotropy
    diff = np.abs(modis_tbl.zonal_slope - modis_tbl.merid_slope)
    sig = np.sqrt(modis_tbl.zonal_slope_err**2 + modis_tbl.merid_slope**2)

    one_sig = diff < 1*sig
    frac = np.sum(one_sig) / len(diff)
    print(f"Fraction within 1 sigma = {frac}")

    # Plot
    fig = plt.figure(figsize=(12, 12))
    plt.clf()

    #ymnx = [-5000., 1000.]

    jg = sns.jointplot(data=modis_tbl, x='zonal_slope', y='merid_slope', 
                       kind='hex', #bins='log', xscale='log',
                       gridsize=100,
                       mincnt=1,
                       marginal_kws=dict(fill=False, 
                                         color='black', bins=100),
                       cmap=plt.get_cmap('YlGnBu')) 
                       #mincnt=1,
    
    jg.ax_joint.set_xlabel(metric_lbls['zonal_slope'])
    jg.ax_joint.set_ylabel(metric_lbls['merid_slope'])
    jg.ax_joint.plot([-5, 1.], [-5, 1.], 'k--')
    #jg.ax_joint.set_ylim(ymnx)

    plotting.set_fontsize(jg.ax_joint, 15.)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_2d_stats(outroot='fig_2dstats_', stat=None, table=None,
                local=False, vmax=None, nbins=40,
                cmap=None, cuts=None, scl = 1, debug=False):
    """ 2D histograms in the UMAP space

    Args:
        outroot (str, optional): [description]. Defaults to 'fig_2dstats_'.
        stat ([type], optional): [description]. Defaults to None.
        local (bool, optional): [description]. Defaults to False.
        vmax ([type], optional): [description]. Defaults to None.
        cmap ([type], optional): [description]. Defaults to None.
        cuts ([type], optional): [description]. Defaults to None.
        scl (int, optional): [description]. Defaults to 1.
        debug (bool, optional): [description]. Defaults to False.
    """

    # Load table
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, cuts=cuts, table=table)

    # Debug?
    if debug:
        modis_tbl = modis_tbl.loc[np.arange(1000000)].copy()

    # Stat
    if stat is None:
        stat = 'min_slope'
    if cmap is None:
        cmap = 'hot'
    outfile = outroot+stat+'.png'

    # Decorate
    outfile = update_outfile(outfile, table)

    # Do it
    median_slope, x_edge, y_edge, ibins = scipy.stats.binned_statistic_2d(
        modis_tbl.U0, modis_tbl.U1, modis_tbl[stat],
        statistic='median', expand_binnumbers=True, bins=[nbins,nbins])

    # Plot
    fig = plt.figure(figsize=(12, 12))
    plt.clf()
    ax = plt.gca()


    cm = plt.get_cmap(cmap)
    mplt = ax.pcolormesh(x_edge, y_edge, 
                     median_slope.transpose(),
                     cmap=cm, 
                     vmax=None) 

    # Color bar
    cbaxes = plt.colorbar(mplt, pad=0., fraction=0.030)
    cbaxes.set_label(f'median({metric_lbls[stat]})', fontsize=17.)
    cbaxes.ax.tick_params(labelsize=15)

    ax.set_xlabel(r'$U_0$')
    ax.set_ylabel(r'$U_1$')

    plotting.set_fontsize(ax, 17.)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_fit_metric(outroot='fig_fit_', metric=None, 
                   local=False, vmax=None, table=None,
                   distr='normal',
                   cmap=None, cuts=None, debug=False):

    # Load table
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, cuts=cuts,
                                               table=table)

    # Debug?
    if debug:
        modis_tbl = modis_tbl.loc[np.arange(1000000)].copy()

    # Stat
    if metric is None:
        metric = 'DT'
    outfile = outroot+metric+'.png'
    # Decorate
    outfile = update_outfile(outfile, table)

    # Fit
    xmnx = modis_tbl[metric].min(), modis_tbl[metric].max()
    xval = np.linspace(xmnx[0], xmnx[1], 1000)
    dx = xval[1]-xval[0]
    if distr == 'normal':
        mean, sigma = stats.norm.fit(modis_tbl[metric])
        vals = stats.norm.pdf(xval, mean, sigma)
        print(f"Gaussian fit: mean={mean}, sigma={sigma}")
    elif distr == 'lognorm':
        shape,loc,scale = stats.lognorm.fit(modis_tbl[metric])
        vals = stats.lognorm.pdf(xval, shape, loc, scale)
        print(f"Log-norm fit: shape={shape}, loc={loc}, scale={scale}")
    else: 
        raise IOError(f"Bad distribution {distr}")

    # Normalize
    sum = dx * np.sum(vals)
    vals /= sum

    # Cumulative
    cumsum = np.cumsum(vals)
    cumsum /= cumsum[-1]
    
    # Plot
    fig = plt.figure(figsize=(10, 5))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # Histogram
    ax_hist = plt.subplot(gs[0])

    _ = sns.histplot(modis_tbl, x=metric, ax=ax_hist,
                     stat='density')
    ax_hist.plot(xval, vals, 'k-')
    


    # CDF
    ax_cdf = plt.subplot(gs[1])
    _ = sns.ecdfplot(modis_tbl, x=metric, ax=ax_cdf)
    ax_cdf.plot(xval, cumsum, 'k--')

    for ax in [ax_hist, ax_cdf]:
        ax.set_xlabel(metric_lbls[metric])
        plotting.set_fontsize(ax, 17.)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

    # KS test
    #embed(header='778 of figs')
    #print(stats.kstest(modis_tbl[metric], distr))


def fig_learn_curve(outfile='fig_learn_curve.png'):
    # Grab the data
    #valid_losses_file = 's3://modis-l2/SSL/models/MODIS_R2019_96/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm/learning_curve/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm_losses_valid.h5'
    valid_losses_file = 's3://modis-l2/SSL/models/MODIS_R2019_96/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm/learning_curve/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm_losses_valid.h5'
    with ulmo_io.open(valid_losses_file, 'rb') as f:
        valid_hf = h5py.File(f, 'r')
    loss_avg_valid = valid_hf['loss_avg_valid'][:]
    loss_step_valid = valid_hf['loss_step_valid'][:]
    loss_valid = valid_hf['loss_valid'][:]
    valid_hf.close()

    #train_losses_file = 's3://modis-l2/SSL/models/MODIS_R2019_96/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm/learning_curve/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm_losses_train.h5'
    train_losses_file = 's3://modis-l2/SSL/models/MODIS_R2019_96/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm/learning_curve/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm_losses_train.h5'
    with ulmo_io.open(train_losses_file, 'rb') as f:
        train_hf = h5py.File(f, 'r')
    loss_train = train_hf['loss_train'][:]
    train_hf.close()

    # Plot
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    ax.plot(loss_valid, label='valid')
    ax.plot(loss_train, c='red', label='train')

    ax.legend(fontsize=15.)

    # Label
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    plotting.set_fontsize(ax, 17.)
    
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_DT_vs_U0(outfile='fig_DT_vs_U0.png',
                 local=None, table=None, nbins=40):
    # Grab the data
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, table=table)

    median, x_edge, y_edge, ibins = scipy.stats.binned_statistic_2d(
        modis_tbl.U0, modis_tbl.U1, modis_tbl['DT'],
        statistic='median', expand_binnumbers=True, bins=[nbins,1])

    xvals = []
    for kk in range(len(x_edge)-1):
        xvals.append(np.mean(x_edge[kk:kk+2]))
        
    # Plot
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    ax.plot(xvals, median.flatten(), 'o')

    #ax.legend(fontsize=15.)

    # Label
    ax.set_xlabel("U0")
    ax.set_ylabel("Median DT")

    plotting.set_fontsize(ax, 17.)
    
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

        
#### ########################## #########################
def main(pargs):

    # UMAP gallery
    if pargs.figure == 'augment':
        fig_augmenting()

    # UMAP LL
    if pargs.figure == 'umap_LL':
        # LL
        fig_umap_colored(local=pargs.local, table=pargs.table,
                         umap_dim=pargs.umap_dim,
                         umap_comp=pargs.umap_comp)

    # UMAP_DT 
    if pargs.figure in 'umap_DT':
        if 'all' in pargs.table:
            metric = 'logDT'
        else:
            metric = 'DT'
        fig_umap_colored(local=pargs.local, table=pargs.table,
                         metric=metric, outfile='fig_umap_DT.png',
                         vmnx=(None,None),
                         umap_dim=pargs.umap_dim,
                         umap_comp=pargs.umap_comp)
        # Clouds
        #fig_umap_colored(local=pargs.local, metric='clouds', outfile='fig_umap_clouds.png',
        #                 vmnx=(None,None))

    # UMAP_slope
    if pargs.figure == 'umap_slope':
        fig_umap_colored(local=pargs.local, table=pargs.table,
                         metric='slope', 
                         outfile='fig_umap_slope.png',
                         cmap='viridis',
                         #vmnx=(-3., -1),
                         hist_param=dict(
                             binx=np.linspace(2,12.5,30),
                             biny=np.linspace(-0.5,9,30)),
                         maxN=400000,
                         umap_dim=pargs.umap_dim,
                         umap_comp=pargs.umap_comp)

    # UMAP gallery
    if pargs.figure == 'umap_gallery':
        #fig_umap_gallery(debug=pargs.debug, in_vmnx=(-5.,5.), table=pargs.table) 
        #fig_umap_gallery(debug=pargs.debug, in_vmnx=None, table=pargs.table,
        #                 outfile='fig_umap_gallery_novmnx.png')
        if pargs.vmnx is not None:
            vmnx = [float(ivmnx) for ivmnx in pargs.vmnx.split(',')]
        else:
            vmnx = [-1,1]
        if pargs.outfile is not None:
            outfile = pargs.outfile
        else:
            outfile = 'fig_umap_gallery.png'
        fig_umap_gallery(debug=pargs.debug, in_vmnx=vmnx, 
                         table=pargs.table ,
            local=pargs.local, outfile=outfile,
            umap_dim=pargs.umap_dim,
            umap_comp=pargs.umap_comp)

    if pargs.figure == 'umap_density':
        fig_umap_density(
            debug=pargs.debug, 
            table=pargs.table ,
            local=pargs.local,
            umap_dim=pargs.umap_dim,
            umap_comp=pargs.umap_comp)

    if pargs.figure == 'umap_geo':
        # Most boring
        #fig_umap_geo('fig_umap_geo_DT0_5656.png',
        #    '96_DT0', [[5.5,6.5], [5.3,6.3]], 
        #    debug=pargs.debug, local=pargs.local)

        # Turbulent in DT1
        #fig_umap_geo('fig_umap_geo_DT1_57n10.png',
        #    '96_DT1', [[5.5,7.0], [-1,-0.25]], 
        #    debug=pargs.debug, local=pargs.local)

        # 'Turbulent' region
        #fig_umap_geo('fig_umap_geo_DT15_7834.png',
        #    '96_DT15', [[7,8], [3,4]], 
        #    debug=pargs.debug, local=pargs.local)
        # Another 'Turbulent' region
        #fig_umap_geo('fig_umap_geo_DT15_8923.png',
        #    '96_DT15', [[8,9], [2,3]], 
        #    debug=pargs.debug, local=pargs.local)

        # Gradient region
        fig_umap_geo('fig_umap_geo_DT15_6779.png',
            '96_DT15', [[5,7], [7.5,9]], 
            debug=pargs.debug, local=pargs.local)

        # 'Turbulent' in DT2
        #fig_umap_geo('fig_umap_geo_DT2_5789.png',
        #    '96_DT2', [[5.5,7], [8.7,9.5]], 
        #    debug=pargs.debug, local=pargs.local)

    if pargs.figure == 'geo_umap':
        # Mediterranean
        #fig_geo_umap('fig_geo_umap_DT15_med.png',
        #    [[0, 60.],   # E
        #     [30, 45.]], # North
        #    debug=pargs.debug, local=pargs.local)

        # Equatorial Pacific
        #fig_geo_umap('fig_geo_umap_DT15_eqpacific.png',
        #    [[-140, -90.],   # W
        #     [-10, 10.]],    # Equitorial 
        #    debug=pargs.debug, local=pargs.local)

        # Coastal California
        #fig_geo_umap('fig_geo_umap_DT15_california.png',
        #    [[-130, -110.],   # W (Pretty crude)
        #     [30, 50.]],      # N
        #    debug=pargs.debug, local=pargs.local)

        # South Atlantic
        #fig_geo_umap('fig_geo_umap_DT1_southatlantic.png',
        #    [[-40, 0.],   # W (Pretty crude)
        #     [-20, -10.]],      # N
        #    table='96_DT1',
        #    debug=pargs.debug, local=pargs.local)

        # Bay of Bengal
        fig_geo_umap('fig_geo_umap_DT15_baybengal.png', 
                     'baybengal',
            table='96_DT15',
            debug=pargs.debug, local=pargs.local)

        # South Pacific
        #fig_geo_umap('fig_geo_umap_DT1_southpacific.png',
        #    [[-120, -90.],   # W (Pretty crude)
        #     [-30, -10.]],      # S
        #    table='96_DT1',
        #    debug=pargs.debug, local=pargs.local)

    if pargs.figure == 'yearly_geo':
        # Equatorial Pacific
        #fig_yearly_geo_umap('fig_yearly_geo_DT15_eqpacific.png',
        #    'eqpacific', rtio_cut=1.5,
        #    debug=pargs.debug, local=pargs.local)

        # Med
        #fig_yearly_geo_umap('fig_yearly_geo_DT15_med.png',
        #    'med', rtio_cut=1.25,
        #    debug=pargs.debug, local=pargs.local)

        # Global using Med
        #fig_yearly_geo_umap('fig_yearly_geo_DT15_global_med.png',
        #    'global', rtio_cut=1.25, rtio_region='med',
        #    debug=pargs.debug, local=pargs.local)

        # Bay of Bengal
        fig_yearly_geo_umap('fig_yearly_geo_DT1_baybengal.png',
            'baybengal', rtio_cut=1.5, table='96_DT1',
            debug=pargs.debug, local=pargs.local)

        # Global using Equatorial
        #fig_yearly_geo_umap('fig_yearly_geo_DT15_global_eqpac.png',
        #    'global', rtio_cut=1.5, rtio_region='eqpacific',
        #    debug=pargs.debug, local=pargs.local)

    if pargs.figure == 'seasonal_geo':
        # Med
        #fig_seasonal_geo_umap('fig_seasonal_geo_DT15_med.png',
        #    'med', rtio_cut=1.25,
        #    debug=pargs.debug, local=pargs.local)

        # Equatorial Pacific
        #fig_seasonal_geo_umap('fig_seasonal_geo_DT15_eqpacific.png',
        #    'eqpacific',
        #    debug=pargs.debug, local=pargs.local)

        # Bay of Bengal
        fig_seasonal_geo_umap('fig_seasonal_geo_DT1_baybengal.png',
            'baybengal', rtio_cut=1.5, table='96_DT1',
            debug=pargs.debug, local=pargs.local)


    # UMAP LL Brazil
    if pargs.figure  == 'umap_brazil':
        fig_umap_colored(outfile='fig_umap_brazil.png', 
                    region='brazil', table=pargs.table,
                    percentiles=(10,90),
                    local=pargs.local,
                    cmap=pargs.cmap,
                    point_size=1., 
                    lbl=r'Brazil, $\Delta T \approx 2$K',
                    vmnx=(-400, 400))

    # UMAP LL Gulf Stream
    if pargs.figure  == 'umap_GS':
        fig_umap_colored(outfile='fig_umap_GS.png', 
                    table=pargs.table,
                    region='GS',
                    local=pargs.local,
                    point_size=1., 
                    lbl=r'Gulf Stream')#, vmnx=(-400, 400))

    # UMAP LL Mediterranean
    if pargs.figure  == 'umap_Med':
        fig_umap_colored(outfile='fig_umap_Med.png', 
                    table=pargs.table,
                    region='Med',
                    local=pargs.local,
                    point_size=1., 
                    lbl=r'Mediterranean')#, vmnx=(-400, 400))
        #fig_umap_2dhist(outfile='fig_umap_2dhist_Med.png', 
        #                cmap='Reds',
        #           table=pargs.table,
        #                local=pargs.local,
        #                region='Med')

    # UMAP 2d Histogram
    if pargs.figure == 'umap_2dhist':
        #
        fig_umap_2dhist(vmax=None, local=pargs.local,
                        table=pargs.table, scl=2)
        # Near norm
        #fig_umap_2dhist(outfile='fig_umap_2dhist_inliers.png',
        #                local=pargs.local, cmap='Greens', 
        #                cuts='inliers')

    # LL vs DT
    if pargs.figure == 'LLvsDT':
        fig_LLvsDT(local=pargs.local, debug=pargs.debug,
                   table=pargs.table)
    
    # slopts
    if pargs.figure == 'slopes':
        fig_slopes(local=pargs.local, debug=pargs.debug,
                    table=pargs.table)

    # Slope vs DT
    if pargs.figure == 'slopevsDT':
        fig_slopevsDT(local=pargs.local, debug=pargs.debug,
                    table=pargs.table)
    
    # 2D Stats
    if pargs.figure == '2d_stats':
        fig_2d_stats(local=pargs.local, debug=pargs.debug,
                    stat=pargs.metric, cmap=pargs.cmap,
                    table=pargs.table)

    # Fit a given metric
    if pargs.figure == 'fit_metric':
        fig_fit_metric(local=pargs.local, debug=pargs.debug,
                       metric=pargs.metric,
                       distr=pargs.distr)
    # learning_curve
    if pargs.figure == 'learning_curve':
        fig_learn_curve()

    # DT vs. U0
    if pargs.figure == 'DT_vs_U0':
        fig_DT_vs_U0(local=pargs.local, table=pargs.table)


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
    parser.add_argument('--umap_dim', type=int, default=2, help="UMAP embedding dimensions")
    parser.add_argument('--umap_comp', type=str, default='0,1', help="UMAP embedding dimensions")
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


# UMAP colored by LL -- python py/fig_ssl_modis.py umap_LL --local
# Slopes -- python py/fig_ssl_modis.py slopes --local
# Slope vs DT -- python py/fig_ssl_modis.py slopevsDT --local
# lowDT 2dstat -- python py/fig_ssl_modis.py 2d_stats --metric lowDT --local
# UMAP of Brazil + 2K -- python py/fig_ssl_modis.py umap_brazil --local
# UMAP of Gulf Stream -- python py/fig_ssl_modis.py umap_GS --local

# UMAP of Med -- python py/fig_ssl_modis.py umap_Med --local

# Simple stats
# DT -- python py/fig_ssl_modis.py fit_metric --metric DT --distr lognorm --local


# ###########################################################
# Cloud free
# UMAP colored by LL -- python py/fig_ssl_modis.py umap_LL --local --table CF
# UMAP colored by DT -- python py/fig_ssl_modis.py umap_DT --local --table CF
# UMAP gallery -- python py/fig_ssl_modis.py umap_gallery --local --table CF
# UMAP of Brazil + 2K -- python py/fig_ssl_modis.py umap_brazil --local --table CF
# UMAP of Med -- python py/fig_ssl_modis.py umap_Med --local --table CF
# UMAP of Gulf Stream -- python py/fig_ssl_modis.py umap_GS --local --table CF
# slope 2dstat -- python py/fig_ssl_modis.py 2d_stats --local --table CF

# 2dhist + contours -- python py/fig_ssl_modis.py umap_2dhist --local --table CF
# DT vs. U0 -- python py/fig_ssl_modis.py DT_vs_U0 --local --table CF

# LL vs DT -- python py/fig_ssl_modis.py LLvsDT --local --table CF

# ################
# UMAP ndim=3

# UMAP colored -- python py/fig_ssl_modis.py umap_LL --local --table CF --umap_dim 3
# UMAP colored 1,2 -- python py/fig_ssl_modis.py umap_LL --local --table CF --umap_dim 3 --umap_comp 1,2

# UMAP gallery 0,1 -- python py/fig_ssl_modis.py umap_gallery --local --table CF --umap_dim 3 
# UMAP gallery 1,2 -- python py/fig_ssl_modis.py umap_gallery --local --table CF --umap_dim 3 --umap_comp 1,2

# ###########################################################
# DT2

# UMAP colored by LL -- python py/fig_ssl_modis.py umap_LL --local --table CF_DT2
# UMAP gallery -- python py/fig_ssl_modis.py umap_gallery --local --table CF_DT2
# slope 2dstat -- python py/fig_ssl_modis.py 2d_stats --local --table CF_DT2

# ###########################################################
# DT15
# UMAP colored by DT -- python py/fig_ssl_modis.py umap_DT --local --table CF_DT15 --umap_comp S0,S1
# UMAP gallery -- python py/fig_ssl_modis.py umap_gallery --local --table CF_DT15 --umap_comp S0,S1 --vmnx=-2,2

# ###########################################################
# DT1
# UMAP colored by LL -- python py/fig_ssl_modis.py umap_LL --local --table CF_DT1
# UMAP gallery -- python py/fig_ssl_modis.py umap_gallery --local --table CF_DT1
# slope 2dstat -- python py/fig_ssl_modis.py 2d_stats --local --table CF_DT1 --cmap jet

# UMAP gallery DT>2 -- python py/fig_ssl_modis.py umap_gallery --local --table CF_DT1_DT2 --umap_comp 0,DT1,1 --vmnx=-2,2 --outfile fig_gallery_vmnx2.png

# ###########################################################
# DT0

# UMAP colored by LL -- python py/fig_ssl_modis.py umap_LL --local --table CF_DT0 --umap_comp S0,S1
# UMAP gallery -- python py/fig_ssl_modis.py umap_gallery --local --table CF_DT0 --umap_comp S0,S1 --vmnx=-1,1

# TODO
# 1) Run cloudy images through new model and UMAP
    # 2) Plot DT vs. U0
# 3) Contours on everything?
# 4) Remake novmnx with 10/90%
    # 5) Spectral maps
# 6) Tracking
# 7) UMAP on DT=2K subset.
# 8) Focus on the arc?




# UMAP colored by DT -- python py/fig_ssl_modis.py umap_DT --local --table CF
# UMAP gallery -- python py/fig_ssl_modis.py umap_gallery --local --table CF
# UMAP of Brazil + 2K -- python py/fig_ssl_modis.py umap_brazil --local --table CF
# UMAP of Med -- python py/fig_ssl_modis.py umap_Med --local --table CF
# UMAP of Gulf Stream -- python py/fig_ssl_modis.py umap_GS --local --table CF
# slope 2dstat -- python py/fig_ssl_modis.py 2d_stats --local --table CF

# 2dhist + contours -- python py/fig_ssl_modis.py umap_2dhist --local --table CF
# DT vs. U0 -- python py/fig_ssl_modis.py DT_vs_U0 --local --table CF



# ###########################################################
# 96% CF
# LL vs DT -- python py/fig_ssl_modis.py LLvsDT --local --table 96

# DT -- python py/fig_ssl_modis.py fit_metric --metric DT --distr lognorm --local --table 96
# LL -- python py/fig_ssl_modis.py fit_metric --metric LL --distr normal --local --table 96
# zonal_slope -- python py/fig_ssl_modis.py fit_metric --metric zonal_slope --distr normal --local --table 96

# Slopes -- python py/fig_ssl_modis.py slopes --local --table 96
# Slope vs DT -- python py/fig_ssl_modis.py slopevsDT --local --table 96

# UMAP colored by LL -- python py/fig_ssl_modis.py umap_LL --local --table 96

# UMAP DT15 colored by DT (all) -- python py/fig_ssl_modis.py umap_DT --local --table 96_DTall --umap_comp S0,S1
# UMAP DT15 colored by DT -- python py/fig_ssl_modis.py umap_DT --local --table 96_DT15 --umap_comp S0,S1
# UMAP DT15 histogram colored by min_slope -- python py/fig_ssl_modis.py umap_slope --local --table 96_DT15 --umap_comp S0,S1

# UMAP gallery -- 
#  python py/fig_ssl_modis.py umap_gallery --local --table 96_DTall --umap_comp S0,S1 --vmnx=-1,1
#  python py/fig_ssl_modis.py umap_gallery --local --table 96_DT0 --umap_comp S0,S1 --vmnx=-0.3, 0.3
#  python py/fig_ssl_modis.py umap_gallery --local --table 96_DT1 --umap_comp S0,S1 --vmnx=-0.75,0.75
#  python py/fig_ssl_modis.py umap_gallery --local --table 96_DT15 --umap_comp S0,S1 --vmnx=-1,1
#  python py/fig_ssl_modis.py umap_gallery --local --table 96_DT2 --umap_comp S0,S1 --vmnx=-1.5,1.5
#  python py/fig_ssl_modis.py umap_gallery --local --table 96_DT4 --umap_comp S0,S1 --vmnx=-2,2
#  python py/fig_ssl_modis.py umap_gallery --local --table 96_DT5 --umap_comp S0,S1 --vmnx=-3,3

# UMAP density -- 
#  python py/fig_ssl_modis.py umap_density --local --table 96_DT0 --umap_comp S0,S1 
#  python py/fig_ssl_modis.py umap_density --local --table 96_DT1 --umap_comp S0,S1 
#  python py/fig_ssl_modis.py umap_density --local --table 96_DT15 --umap_comp S0,S1 
#  python py/fig_ssl_modis.py umap_density --local --table 96_DT2 --umap_comp S0,S1 
#  python py/fig_ssl_modis.py umap_density --local --table 96_DT4 --umap_comp S0,S1 
#  python py/fig_ssl_modis.py umap_density --local --table 96_DT5 --umap_comp S0,S1 

# UMAP geographic
#  python py/fig_ssl_modis.py umap_geo --local 
#  python py/fig_ssl_modis.py geo_umap --local 

# UMAP temporal
#  python py/fig_ssl_modis.py yearly_geo --local 
#  python py/fig_ssl_modis.py seasonal_geo --local 

# MISC
#  python py/fig_ssl_modis.py learning_curve
#  python py/fig_ssl_modis.py augment
