""" Figures for SSL paper on MODIS """
from datetime import datetime
import os, sys
import numpy as np
from urllib.parse import urlparse
import datetime

import scipy

import healpy as hp
import pandas

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.dates as mdates

mpl.rcParams['font.family'] = 'stixgeneral'

import seaborn as sns

import h5py

from ulmo import plotting
from ulmo.utils import utils as utils
from ulmo.utils import table as table_utils
from ulmo.utils import image_utils
from ulmo.nenya import nenya_umap


from IPython import embed



def umap_multi_metric(tbl, 
                binx:np.ndarray,
                biny:np.ndarray,
                stat='median', 
                cmap=None,
                umap_keys=['US0','US1'],
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
        elif 'FS_Npos' in metric: 
            istat = 'mean'
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
            if 'log10' in metric:
                stat2d = np.log10(stat2d)
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



def umap_gallery(tbl:pandas.DataFrame, outfile:str,
                     local:str=None, in_vmnx=None,
                     umap_comp='0,1', nxy=16,
                     min_pts=None,
                     umap_dim=2,
                     umap_rngs=None,
                     extra_umap_rngs=None,
                     umap_keys=['US0','US1'],
                     seed=None,
                     annotate=False,
                     use_std_lbls=True,
                     cut_to_inner:int=None,
                     skip_incidence=False,
                     debug=False): 
    """ UMAP gallery

    Args:
        outfile (str): [description]. Defaults to 'fig_umap_LL.png'.
        version (int, optional): [description]. Defaults to 1.
        local (str, optional): Path to the PreProc files
        debug (bool, optional): [description]. Defaults to False.
        cut_to_inner (int, optional): If provided, cut the image
            down to the inner npix x npix with npix = cut_to_inner

    Raises:
        IOError: [description]
    """
    if min_pts is None: 
        min_pts = 10
    # Seed
    if seed is not None:
        np.random.seed(seed)

    if debug:
        nxy = 4

    # Cut table
    dxv = 0.5
    dyv = 0.25
    umap_grid = nenya_umap.grid_umap(tbl[umap_keys[0]].values,
                            tbl[umap_keys[1]].values, nxy=nxy)
    # Unpack
    xmin, xmax = umap_grid['xmin'], umap_grid['xmax']
    ymin, ymax = umap_grid['ymin'], umap_grid['ymax']
    dxv = umap_grid['dxv']
    dyv = umap_grid['dyv']


    # cut
    good = (tbl[umap_keys[0]] > xmin) & (
        tbl[umap_keys[0]] < xmax) & (
        tbl[umap_keys[1]] > ymin) & (
            tbl[umap_keys[1]] < ymax) 
    if 'LL' in tbl.keys():
        good &= np.isfinite(tbl.LL)

    tbl = tbl.loc[good].copy()
    num_samples = len(tbl)
    print(f"We have {num_samples} making the cuts.")

    if debug: # take a subset
        print("DEBUGGING IS ON")
        nsub = 500000
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        idx = idx[0:nsub]
        tbl = tbl.iloc[idx].copy()

    # Fig
    _, cm = plotting.load_palette()
    fsz = 15.
    if annotate or skip_incidence:
        fsize = (9,8)
    else:
        fsize = (12,8)
    fig = plt.figure(figsize=fsize)
    plt.clf()

    if annotate:
        ax_gallery = fig.add_axes([0.10, 0.12, 0.75, 0.85])
    elif skip_incidence:
        ax_gallery = fig.add_axes([0.10, 0.12, 0.75, 0.85])
    else:
        ax_gallery = fig.add_axes([0.05, 0.1, 0.6, 0.90])

    if use_std_lbls:
        ax_gallery.set_xlabel(r'$U_0$')
        ax_gallery.set_ylabel(r'$U_1$')
    else:
        ax_gallery.set_xlabel(r'$'+umap_keys[0]+'$')
        ax_gallery.set_ylabel(r'$'+umap_keys[1]+'$')

    # Gallery
    #dxdy=(0.3, 0.3)
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
            pts_cut = (tbl[umap_keys[0]] >= x) & (
                tbl[umap_keys[0]] < x+dxv) & (
                tbl[umap_keys[1]] >= y) & (tbl[umap_keys[1]] < y+dxv)
            if 'LL' in tbl.keys():
                pts_cut &= np.isfinite(tbl.LL)
            pts = np.where(pts_cut)[0]
            #pts = np.where((tbl[umap_keys[0]] >= x) & (
            #    tbl[umap_keys[0]] < x+dxv) & (
            #    tbl[umap_keys[1]] >= y) & (tbl[umap_keys[1]] < y+dxv)
            #               & np.isfinite(tbl.LL))[0]
            if len(pts) < min_pts:
                continue

            # Pick a random one
            ichoice = np.random.choice(len(pts), size=1)
            idx = int(pts[ichoice])
            cutout = tbl.iloc[idx]

            # Image
            axins = ax_gallery.inset_axes(
                    [x, y, 0.9*dxv, 0.9*dyv], 
                    transform=ax_gallery.transData)
            # Load
            try:
                if local is not None:
                    parsed_s3 = urlparse(cutout.pp_file)
                    local_file = os.path.join(local, parsed_s3.path[1:])
                    cutout_img = image_utils.grab_image(
                        cutout, close=True, local_file=local_file)
                else:
                    cutout_img = image_utils.grab_image(cutout, close=True)
            except:
                embed(header='290 of nenya.figures')                                                    
            # Cut down?
            if cut_to_inner is not None:
                imsize = cutout_img.shape[0]
                x0, y0 = [imsize//2-cut_to_inner//2]*2
                x1, y1 = [imsize//2+cut_to_inner//2]*2
                cutout_img = cutout_img[x0:x1,y0:y1]
            # Limits
            if in_vmnx is None:
                imin, imax = cutout_img.min(), cutout_img.max()
                amax = max(np.abs(imin), np.abs(imax))
                vmnx = (-1*amax, amax)
            elif in_vmnx[0] == -999:
                DT = cutout.T90 - cutout.T10
                vmnx = (-1*DT, DT)
            else: 
                vmnx = in_vmnx
            # Plot
            sns_ax = sns.heatmap(np.flipud(cutout_img), 
                            xticklabels=[], 
                     vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap=cm, cbar=plt_cbar,
                     cbar_ax=ax_cbar, cbar_kws=cbar_kws,
                     ax=axins)
            sns_ax.set_aspect('equal', 'datalim')
            # Only do this once
            if plt_cbar:
                plt_cbar = False
            ndone += 1
            print(f'ndone= {ndone}')#, LL={cutout.LL}')
            if ndone > nmax:
                break
        if ndone > nmax:
            break

    plotting.set_fontsize(ax_gallery, fsz)
    #ax.set_aspect('equal', 'datalim')
    #ax.set_aspect('equal')#, 'datalim')

    '''
    # Box?
    if umap_rngs is not None:
        umap_rngs = parse_umap_rngs(umap_rngs)
            # Create patch collection with specified colour/alpha
        rect = Rectangle((umap_rngs[0][0], umap_rngs[1][0]),
            umap_rngs[0][1]-umap_rngs[0][0],
            umap_rngs[1][1]-umap_rngs[1][0],
            linewidth=2, edgecolor='k', facecolor='none', ls='-',
            zorder=10)
        ax_gallery.add_patch(rect)

    # Another?
    if extra_umap_rngs is not None:
        umap_rngs = parse_umap_rngs(extra_umap_rngs)
            # Create patch collection with specified colour/alpha
        rect2 = Rectangle((umap_rngs[0][0], umap_rngs[1][0]),
            umap_rngs[0][1]-umap_rngs[0][0],
            umap_rngs[1][1]-umap_rngs[1][0],
            linewidth=2, edgecolor='k', facecolor='none', ls='--',
            zorder=10)
        ax_gallery.add_patch(rect2)
    '''

    # Incidence plot
    if not annotate and not skip_incidence:
        ax_incidence = fig.add_axes([0.71, 0.45, 0.25, 0.36])

        umap_density(tbl, umap_keys,
                     umap_grid=umap_grid, umap_comp=umap_comp,
                     show_cbar=True, ax=ax_incidence, fsz=12.)
    #ax_incidence.plot(np.arange(10), np.arange(10))

    #plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))



def umap_density(tbl:pandas.DataFrame, umap_keys:list,
                 outfile:str=None, 
                     local=False, table='std', 
                     umap_comp='0,1', umap_grid=None,
                     umap_dim=2, cmap=None, nxy=16,
                     fsz=19.,
                     use_std_lbls=True,
                     show_cbar = False,
                     debug=False, ax=None): 

    # Boundaries of the box
    if umap_grid is None:
        umap_grid = nenya_umap.grid_umap(tbl[umap_keys[0]].values, tbl[umap_keys[0]].values,
                  nxy=nxy)

    xmin, xmax = umap_grid['xmin'], umap_grid['xmax']
    ymin, ymax = umap_grid['ymin'], umap_grid['ymax']
    dxv = umap_grid['dxv']
    dyv = umap_grid['dyv']

    # Grid
    xval = np.arange(xmin, xmax+dxv, dxv)
    yval = np.arange(ymin, ymax+dyv, dyv)

    # cut
    good = (tbl[umap_keys[0]] > xmin) & (
        tbl[umap_keys[0]] < xmax) & (
        tbl[umap_keys[1]] > ymin) & (
            tbl[umap_keys[1]] < ymax) 
    if 'LL' in tbl.keys():
        good += np.isfinite(tbl.LL)

    tbl = tbl.loc[good].copy()
    num_samples = len(tbl)
    print(f"We have {num_samples} making the cuts.")

    counts, xedges, yedges = np.histogram2d(
        tbl[umap_keys[0]], 
        tbl[umap_keys[1]], bins=(xval, yval))

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
    