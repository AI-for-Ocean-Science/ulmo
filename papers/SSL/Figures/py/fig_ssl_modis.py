""" Figures for SSL paper on MODIS """
import os, sys
from typing import IO
import numpy as np
import scipy
from scipy import stats

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


metric_lbls = dict(min_slope=r'$\alpha_{\rm min}$',
                   clear_fraction='1-CC',
                   DT=r'$\Delta T$',
                   lowDT=r'$\Delta T_{\rm low}$',
                   absDT=r'$|T_{90}| - |T_{10}|$',
                   LL='LL',
                   zonal_slope=r'$\alpha_{\rm AS}}$',
                   merid_slope=r'$\alpha_{\rm AT}}$',
                   )

if os.getenv('SST_OOD'):
    local_modis_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_L2_std.parquet')



def load_modis_tbl(tbl_file=None, local=False, cuts=None,
                   region=None):
    if tbl_file is None:
        tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    if local:
        tbl_file = local_modis_file

    # Load
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # DT
    if 'DT' not in modis_tbl.keys():
        modis_tbl['DT'] = modis_tbl.T90 - modis_tbl.T10
    modis_tbl['logDT'] = np.log10(modis_tbl.DT)
    modis_tbl['lowDT'] = modis_tbl.mean_temperature - modis_tbl.T10
    modis_tbl['absDT'] = np.abs(modis_tbl.T90) - np.abs(modis_tbl.T10)

    # Slopes
    modis_tbl['min_slope'] = np.minimum(
        modis_tbl.zonal_slope, modis_tbl.merid_slope)

    # Cut
    goodLL = np.isfinite(modis_tbl.LL)
    if cuts is None:
        good = goodLL
    elif cuts == 'inliers':
        inliers = (modis_tbl.LL > 200.) & (modis_tbl.LL < 400)
        good = goodLL & inliers
    modis_tbl = modis_tbl[good].copy()

    # Region?
    if region is None:
        pass
    elif region == 'brazil':
        # Brazil
        in_brazil = ((np.abs(modis_tbl.lon.values + 57.5) < 10.)  & 
            (np.abs(modis_tbl.lat.values + 43.0) < 10))
        in_DT = np.abs(modis_tbl.DT - 2.05) < 0.05
        modis_tbl = modis_tbl[in_brazil & in_DT].copy()
    elif region == 'GS':
        # Gulf Stream
        in_GS = ((np.abs(modis_tbl.lon.values + 69.) < 3.)  & 
            (np.abs(modis_tbl.lat.values - 39.0) < 1))
        modis_tbl = modis_tbl[in_GS].copy()
    elif region == 'Med':
        # Mediterranean
        in_Med = ((modis_tbl.lon > -5.) & (modis_tbl.lon < 30.) &
            (np.abs(modis_tbl.lat.values - 36.0) < 5))
        modis_tbl = modis_tbl[in_Med].copy()
    else: 
        raise IOError(f"Bad region! {region}")

    return modis_tbl

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
                metric='LL',
                local=False, 
                point_size = None, 
                lbl=None,
                vmnx = (-1000., None),
                region=None,
                debug=False): 
    """ UMAP colored by LL or something else

    Args:
        outfile (str, optional): [description]. Defaults to 'fig_umap_LL.png'.
        local (bool, optional): [description]. Defaults to True.
        debug (bool, optional): [description]. Defaults to False.

    Raises:
        IOError: [description]
    """
    # Load table
    modis_tbl = load_modis_tbl(local=local, cuts=cuts, region=region)
    num_samples = len(modis_tbl)


    if debug: # take a subset
        print("DEBUGGING IS ON")
        nsub = 500000
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        idx = idx[0:nsub]
        modis_tbl = modis_tbl.loc[idx].copy()

    # Metric
    if metric == 'LL':
        values = modis_tbl.LL 
    elif metric == 'DT':
        values = np.log10(modis_tbl.DT.values)
    elif metric == 'clouds':
        values = modis_tbl.clear_fraction
    else:
        raise IOError("Bad metric!")
    

    # Start the figure
    fig = plt.figure(figsize=(8, 8))
    plt.clf()
    gs = gridspec.GridSpec(1, 1)

    # Just the UMAP colored by LL
    ax0 = plt.subplot(gs[0])

    if point_size is None:
        point_size = 1. / np.sqrt(num_samples)
    img = ax0.scatter(modis_tbl.U0, modis_tbl.U1,
            s=point_size, c=values,
            cmap='jet', vmin=vmnx[0], vmax=vmnx[1])
    cb = plt.colorbar(img, pad=0., fraction=0.030)
    cb.set_label(metric, fontsize=14.)
    #
    ax0.set_xlabel(r'$U_0$')
    ax0.set_ylabel(r'$U_1$')
    #ax0.set_aspect('equal')#, 'datalim')

    fsz = 17.
    plotting.set_fontsize(ax0, fsz)

    # Set boundaries
    #xmin, xmax = modis_tbl.U0.min()-dxdy[0], modis_tbl.U0.max()+dxdy[0]
    #ymin, ymax = modis_tbl.U1.min()-dxdy[1], modis_tbl.U1.max()+dxdy[1]
    '''
    xmin, xmax = -4.5, 7
    ymin, ymax = 4.5, 10.5
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)
    '''

    # Label
    if lbl is not None:
        ax0.text(0.05, 0.9, lbl, transform=ax0.transAxes,
              fontsize=15, ha='left', color='k')

    #plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_umap_gallery(outfile='fig_umap_gallery_vmnx5.png',
                     version=1, local=False, 
                     in_vmnx=None,
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
    if version == 1:                    
        tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    else:
        raise IOError("bad version number")
    if local:
        tbl_file = local_modis_file
    # Load
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Cut table
    xmin, xmax = -4.5, 7
    ymin, ymax = 4.5, 10.5
    good = (modis_tbl.U0 > xmin) & (modis_tbl.U0 < xmax) & (
        modis_tbl.U1 > ymin) & (modis_tbl.U1 < ymax) & np.isfinite(modis_tbl.LL)
    modis_tbl = modis_tbl.loc[good].copy()
    num_samples = len(modis_tbl)

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
    fig = plt.figure(figsize=(8, 8))
    plt.clf()
    ax = plt.gca()

    ax.set_xlabel(r'$U_0$')
    ax.set_ylabel(r'$U_1$')

    # Gallery
    #dxdy=(0.3, 0.3)
    #xmin, xmax = modis_tbl.U0.min()-dxdy[0], modis_tbl.U0.max()+dxdy[0]
    #ymin, ymax = modis_tbl.U1.min()-dxdy[1], modis_tbl.U1.max()+dxdy[1]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    print('x,y', xmin, xmax, ymin, ymax)

    
    # ###################
    # Gallery time

    # Grid
    dxv = 0.5
    dyv = 0.25
    xval = np.arange(xmin, xmax+dxv, dxv)
    yval = np.arange(ymin, ymax+dyv, dyv)

    # Ugly for loop
    ndone = 0
    if debug:
        nmax = 100
    else:
        nmax = 1000000000
    for x in xval[:-1]:
        for y in yval[:-1]:
            pts = np.where((modis_tbl.U0 >= x) & (modis_tbl.U0 < x+dxv) & (
                modis_tbl.U1 >= y) & (modis_tbl.U1 < y+dxv)
                           & np.isfinite(modis_tbl.LL))[0]
            if len(pts) == 0:
                continue

            # Pick a random one
            ichoice = np.random.choice(len(pts), size=1)
            idx = int(pts[ichoice])
            cutout = modis_tbl.iloc[idx]

            # Image
            axins = ax.inset_axes(
                    [x, y, 0.9*dxv, 0.9*dyv], 
                    transform=ax.transData)
            try:
                cutout_img = image_utils.grab_image(cutout, close=True)
            except:
                embed(header='198 of plotting')                                                    
            # Limits
            if in_vmnx is not None:
                vmnx = in_vmnx
            else:
                imin, imax = cutout_img.min(), cutout_img.max()
                amax = max(np.abs(imin), np.abs(imax))
                vmnx = (-1*amax, amax)
            # Plot
            _ = sns.heatmap(np.flipud(cutout_img), xticklabels=[], 
                     vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap=cm, cbar=False,
                     ax=axins)
            ndone += 1
            print(f'ndone= {ndone}, LL={cutout.LL}')
            if ndone > nmax:
                break
        if ndone > nmax:
            break

    plotting.set_fontsize(ax, fsz)
    #ax.set_aspect('equal', 'datalim')
    #ax.set_aspect('equal')#, 'datalim')

    #plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_umap_2dhist(outfile='fig_umap_2dhist.png',
                    version=1, local=False, vmax=None, 
                    cmap=None, cuts=None, region=None,
                    scl = 1):

    # Load
    modis_tbl = load_modis_tbl(local=local, cuts=cuts, region=region)

    # 
    xmin, xmax = -4.5, 8
    ymin, ymax = 4.5, 10.5
    # Histogram
    bins_U0 = np.linspace(xmin, xmax, 23*scl)
    bins_U1 = np.linspace(ymin,ymax, 24*scl)
    counts, xedges, yedges = np.histogram2d(modis_tbl.U0, modis_tbl.U1,
                                            bins=(bins_U0, bins_U1))

    fig = plt.figure(figsize=(12, 12))
    plt.clf()
    ax = plt.gca()

    if cmap is None:
        cmap = "Blues"
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


def fig_LLvsDT(outfile='fig_LLvsDT.png', local=False, vmax=None, 
                    cmap=None, cuts=None, scl = 1, debug=False):
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
    modis_tbl = load_modis_tbl(local=local, cuts=cuts)

    # Debug?
    if debug:
        modis_tbl = modis_tbl.loc[np.arange(1000000)].copy()

    # Plot
    fig = plt.figure(figsize=(12, 12))
    plt.clf()

    ymnx = [-5000., 1000.]

    jg = sns.jointplot(data=modis_tbl, x='DT', y='LL', kind='hex',
                       bins='log', gridsize=250, xscale='log',
                       cmap=plt.get_cmap('autumn'), mincnt=1,
                       marginal_kws=dict(fill=False, color='black', bins=100)) 
    jg.ax_joint.set_xlabel(r'$\Delta T$')
    jg.ax_joint.set_ylim(ymnx)

    plotting.set_fontsize(jg.ax_joint, 15.)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_slopevsDT(outfile='fig_slopevsDT.png', local=False, vmax=None, 
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
    modis_tbl = load_modis_tbl(local=local, cuts=cuts)

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


def fig_slopes(outfile='fig_slopes.png', local=False, vmax=None, 
                    cmap=None, cuts=None, scl = 1, debug=False):

    # Load table
    modis_tbl = load_modis_tbl(local=local, cuts=cuts)

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


def fig_2d_stats(outroot='fig_2dstats_', stat=None,
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
    modis_tbl = load_modis_tbl(local=local, cuts=cuts)

    # Debug?
    if debug:
        modis_tbl = modis_tbl.loc[np.arange(1000000)].copy()

    # Stat
    if stat is None:
        stat = 'min_slope'
    if cmap is None:
        cmap = 'hot'
    outfile = outroot+stat+'.png'

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
                   local=False, vmax=None, 
                   distr='normal',
                   cmap=None, cuts=None, debug=False):

    # Load table
    modis_tbl = load_modis_tbl(local=local, cuts=cuts)

    # Debug?
    if debug:
        modis_tbl = modis_tbl.loc[np.arange(1000000)].copy()

    # Stat
    if metric is None:
        metric = 'DT'
    outfile = outroot+metric+'.png'

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
    embed(header='656 of figs')
    print(stats.kstest(modis_tbl[metric], 'lognorm'))


def fig_learn_curve(outfile='fig_learn_curve.png'):
    # Grab the data
    valid_losses_file = 's3://modis-l2/SSL/models/MODIS_R2019_2010/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm/learning_curve/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm_losses_valid.h5'
    with ulmo_io.open(valid_losses_file, 'rb') as f:
        valid_hf = h5py.File(f, 'r')
    loss_avg_valid = valid_hf['loss_avg_valid'][:]
    loss_step_valid = valid_hf['loss_step_valid'][:]
    loss_valid = valid_hf['loss_valid'][:]
    valid_hf.close()

    train_losses_file = 's3://modis-l2/SSL/models/MODIS_R2019_2010/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm/learning_curve/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm_losses_train.h5'
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
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

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
        fig_umap_colored(local=pargs.local)
        # DT
        #fig_umap_colored(local=pargs.local, metric='DT', outfile='fig_umap_DT.png',
        #                 vmnx=(None, None))
        # Clouds
        #fig_umap_colored(local=pargs.local, metric='clouds', outfile='fig_umap_clouds.png',
        #                 vmnx=(None,None))

    # UMAP gallery
    if pargs.figure == 'umap_gallery':
        fig_umap_gallery(debug=pargs.debug, in_vmnx=(-5.,5.)) 
        fig_umap_gallery(debug=pargs.debug, in_vmnx=None,
                         outfile='fig_umap_gallery_novmnx.png')
        fig_umap_gallery(debug=pargs.debug, in_vmnx=(-1.,1.), 
                         outfile='fig_umap_gallery_vmnx1.png')

    # UMAP LL Brazil
    if pargs.figure  == 'umap_brazil':
        fig_umap_colored(outfile='fig_umap_brazil.png', 
                    region='brazil',
                    local=pargs.local,
                    point_size=1., 
                    lbl=r'Brazil, $\Delta T \approx 2$K',
                    vmnx=(-400, 400))

    # UMAP LL Gulf Stream
    if pargs.figure  == 'umap_GS':
        fig_umap_colored(outfile='fig_umap_GS.png', 
                    region='GS',
                    local=pargs.local,
                    point_size=1., 
                    lbl=r'Gulf Stream')#, vmnx=(-400, 400))

    # UMAP LL Mediterranean
    if pargs.figure  == 'umap_Med':
        #fig_umap_colored(outfile='fig_umap_Med.png', 
        #            region='Med',
        #            local=pargs.local,
        #            point_size=1., 
        #            lbl=r'Mediterranean')#, vmnx=(-400, 400))
        fig_umap_2dhist(outfile='fig_umap_2dhist_Med.png', 
                        cmap='Reds',
                        local=pargs.local,
                        region='Med')

    # UMAP 2d Histogram
    if pargs.figure == 'umap_2dhist':
        #
        fig_umap_2dhist(vmax=80000, local=pargs.local)
        # Near norm
        fig_umap_2dhist(outfile='fig_umap_2dhist_inliers.png',
                        local=pargs.local, cmap='Greens', 
                        cuts='inliers')

    # LL vs DT
    if pargs.figure == 'LLvsDT':
        fig_LLvsDT(local=pargs.local, debug=pargs.debug)
    
    # slopts
    if pargs.figure == 'slopes':
        fig_slopes(local=pargs.local, debug=pargs.debug)

    # Slope vs DT
    if pargs.figure == 'slopevsDT':
        fig_slopevsDT(local=pargs.local, debug=pargs.debug)
    
    # 2D Stats
    if pargs.figure == '2d_stats':
        fig_2d_stats(local=pargs.local, debug=pargs.debug,
                    stat=pargs.metric, cmap=pargs.cmap)

    # Fit a given metric
    if pargs.figure == 'fit_metric':
        fig_fit_metric(local=pargs.local, debug=pargs.debug,
                       metric=pargs.metric,
                       distr=pargs.distr)
    # learning_curve
    if pargs.figure == 'learning_curve':
        fig_learn_curve()

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
    parser.add_argument('--distr', type=str, default='normal',
                        help='Distribution to fit [normal, lognorm]')
    parser.add_argument('--local', default=False, action='store_true', 
                        help='Use local file(s)?')
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
