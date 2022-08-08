""" Figures for Bay of Bengal """

from datetime import datetime
from operator import mod
import os, sys
import numpy as np
from requests import head
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
#sys.path.append(os.path.abspath("../Analysis/py"))
#import bob_analy

sys.path.append(os.path.abspath("../../SSL/Analysis/py"))
import ssl_paper_analy

def fig_spatial(outfile='fig_spatial.png', nside=64, local=True,
                    table='96_DTall'):
    """
    Spatial distribution of the cutouts

    Parameters
    ----------
    pproc
    outfile
    nside

    Returns
    -------

    """
    # Load
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, table=table)

    # Cut to speed up
    lons=[60, 120.]   # E
    lats=[0, 30.] # N
    geo = ( (modis_tbl.lon > lons[0]-10) &
        (modis_tbl.lon < lons[1]+10) &
        (modis_tbl.lat > lats[0]) &
        (modis_tbl.lat < lats[1]) )
    modis_tbl = modis_tbl[geo].copy()

    lbl = 'evals'
    use_log = True
    use_mask = True

    # Healpix me
    hp_events, hp_lons, hp_lats = image_utils.evals_to_healpix(
        modis_tbl, nside, log=use_log, mask=use_mask)


   # Figure
    fig = plt.figure(figsize=(12,8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm = plt.get_cmap('Blues')
    # Cut
    good = np.invert(hp_events.mask)
    img = plt.scatter(x=hp_lons[good],
        y=hp_lats[good],
        c=hp_events[good], 
        cmap=cm,
        #vmax=vmax, 
        s=30,
        marker='s',
        transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    lbl = r"$\log_{10} \, N_{\rm cutouts}$"
    if lbl is not None:
        cb.set_label(lbl, fontsize=20.)
    cb.ax.tick_params(labelsize=17)

    # Zoom in

    # Coast lines
    ax.coastlines(zorder=10)
    ax.add_feature(cartopy.feature.LAND, 
        facecolor='gray', edgecolor='black')
    ax.set_extent(lons+lats)
    #ax.set_global()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
        color='black', alpha=0.5, linestyle=':', 
        draw_labels=True)
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


def fig_bob_gallery(outfile='fig_bob_gallery_w_latent.png', 
                    geo_region='baybengal',
                    local=True, 
                    ngallery=8,
                    seed=1234,
                    skip_latent=False,
                    table='96_DTall'):
    # Random seed
    np.random.seed(seed)
    #np.random.RandomState(seed=seed)

    _, cm = plotting.load_palette()
    # Load Table
    if table == '96_DT15':
        print('UPDATE WITH ALL')
        embed(header='59 of fig bob')
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, table=table)

    # Isolate Bay of Bengal
    lons = ssl_paper_analy.geo_regions[geo_region]['lons']
    lats = ssl_paper_analy.geo_regions[geo_region]['lats']
    geo = ( (modis_tbl.lon > lons[0]) &
        (modis_tbl.lon < lons[1]) &
        (modis_tbl.lat > lats[0]) &
        (modis_tbl.lat < lats[1]) )

    modis_cut = modis_tbl[geo].copy()

    # Grab random cutouts by LL distribution
    bins = np.linspace(0., 1., ngallery+1)
    all_idx = []
    for kk in range(bins.size-1):
        LLmin = np.percentile(modis_cut.LL, bins[kk]*100)
        LLmax = np.percentile(modis_cut.LL, bins[kk+1]*100)
        #
        inLL = (modis_cut.LL >= LLmin) & (
            modis_cut.LL < LLmax) & (modis_cut.pp_type == 0)
        # Grab one
        idx = np.random.choice(np.where(inLL)[0], size=1)
        all_idx.append(idx[0])

        
    # Figure
    fig = plt.figure(figsize=(12, 6))
    plt.clf()
    gs = gridspec.GridSpec(2, ngallery//2)

    for ss, idx in enumerate(all_idx):
        ax = plt.subplot(gs[ss])
        row = modis_cut.iloc[idx]
        assert row.pp_type == 0 # valid only
        print(f"UID: {row.UID}")


        # Cutout
        if skip_latent:
            ax_cutout = ax
        else:
            ax_cutout = ax.inset_axes([0, 0.1, 1., 0.9],
                    transform=ax.transData)
        if local:
            pp_file = os.path.join(os.getenv('SST_OOD'),
                             'MODIS_L2', 'PreProc', os.path.basename(row.pp_file))
            img = image_utils.grab_image(row, local_file=pp_file)
        else:
            img = image_utils.grab_image(row)
        _ = sns.heatmap(img, xticklabels=[], 
                     vmin=-1.5, vmax=1.5,
                     yticklabels=[], cmap=cm,
                     cbar=False, ax=ax_cutout)
        ax_cutout.xaxis.set_visible(False)
        ax_cutout.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Plot the latent?
        if not skip_latent:
            pp_base = os.path.basename(row.pp_file)
            lpath = os.path.join(os.getenv('SST_OOD'),
                                'MODIS_L2', 'SSL', 'latents', 
                                'MODIS_R2019_96', 
                                'SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_5_cosine_warm')
            latent_base = pp_base.replace('preproc', 'latents')
            latent_file = os.path.join(lpath, latent_base)

            l_f = h5py.File(latent_file, 'r')
            latent = l_f['valid'][row.pp_idx, :].reshape((1,256))

            ax_latent = ax.inset_axes([0, 0, 1., 0.1],
                        transform=ax.transData)
            _ = sns.heatmap(latent, xticklabels=[], 
                        vmin=-0.4, vmax=0.4,
                        yticklabels=[], cmap='Greys', 
                        cbar=False, ax=ax_latent)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
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

    # Spatial distribution
    if pargs.figure == 'spatial':
        fig_spatial()

    # BoB gallery
    if pargs.figure == 'gallery':
        #fig_bob_gallery(outfile='fig_bob_gallery_wo_latent.png', skip_latent=True)
        fig_bob_gallery()


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

# Figs

# Spatial distribution
# python py/fig_bay_of_bengal.py spatial

# Gallery
# python py/fig_bay_of_bengal.py gallery