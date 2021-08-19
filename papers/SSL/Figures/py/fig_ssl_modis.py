""" Figures for SSL paper on MODIS """
import os, sys
from typing import IO
import numpy as np
import scipy

import argparse

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import h5py
from tqdm.auto import trange
import time

import torch
from ulmo.ssl.util import adjust_learning_rate
from ulmo.ssl.util import set_optimizer

from ulmo.ssl.train_util import Params, option_preprocess
from ulmo.ssl.train_util import modis_loader_v2, set_model
from ulmo.ssl.train_util import train_learn_curve, valid_learn_curve

from ulmo import plotting
from ulmo.utils import utils as utils

from ulmo import io as ulmo_io
from ulmo.ssl import single_image as ssl_simage
from ulmo.utils import image_utils

from IPython import embed

if os.getenv('SST_OOD'):
    local_modis_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_L2_std.parquet')

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("SSL Figures")
    parser.add_argument("figure", type=str, help="function to execute: 'slopes'")
    parser.add_argument('--stat', type=str, help='Stat for the figure')
    parser.add_argument('--local', default=False, action='store_true', 
                        help='Use local file(s)?')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    args = parser.parse_args()
    
    return args


def load_modis_tbl(tbl_file=None, local=False, cuts=None):
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
                metric='LL',
                version=1, local=False, 
                point_size = None, 
                lbl=None,
                vmnx = (-1000., None),
                region=None,
                debug=False): 
    """ UMAP colored by LL

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
    num_samples = len(modis_tbl)
    if 'DT' not in modis_tbl.keys():
        modis_tbl['DT'] = modis_tbl.T90 - modis_tbl.T10

    # Region?
    if region is None:
        pass
    elif region == 'brazil':
            # Add in DT

        # Brazil
        in_brazil = ((np.abs(modis_tbl.lon.values + 57.5) < 10.)  & 
            (np.abs(modis_tbl.lat.values + 43.0) < 10))
        in_DT = np.abs(modis_tbl.DT - 2.05) < 0.05
        modis_tbl = modis_tbl[in_brazil & in_DT].copy()
    

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
    cb.set_label(metric, fontsize=12.)
    #
    ax0.set_xlabel(r'$U_0$')
    ax0.set_ylabel(r'$U_1$')
    #ax0.set_aspect('equal')#, 'datalim')

    fsz = 13.
    set_fontsize(ax0, fsz)

    # Set boundaries
    #xmin, xmax = modis_tbl.U0.min()-dxdy[0], modis_tbl.U0.max()+dxdy[0]
    #ymin, ymax = modis_tbl.U1.min()-dxdy[1], modis_tbl.U1.max()+dxdy[1]
    xmin, xmax = -4.5, 7
    ymin, ymax = 4.5, 10.5
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)

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

    set_fontsize(ax, fsz)
    #ax.set_aspect('equal', 'datalim')
    #ax.set_aspect('equal')#, 'datalim')

    #plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_umap_2dhist(outfile='fig_umap_2dhist.png',
                    version=1, local=False, vmax=None, 
                    cmap=None, cuts=None,
                    scl = 1):

    if version == 1:                    
        tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    else:
        raise IOError("bad version number")
    if local:
        tbl_file = local_modis_file

    # Load
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Cut
    goodLL = np.isfinite(modis_tbl.LL)
    if cuts is None:
        good = goodLL
    elif cuts == 'inliers':
        inliers = (modis_tbl.LL > 200.) & (modis_tbl.LL < 400)
        good = goodLL & inliers
    modis_tbl = modis_tbl[good].copy()

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
                       cmap=plt.get_cmap('winter'), mincnt=1,
                       marginal_kws=dict(fill=False, color='black', bins=100)) 
    jg.ax_joint.set_xlabel(r'$\Delta T$')
    jg.ax_joint.set_ylim(ymnx)

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
                       cmap=plt.get_cmap('OrRd')) 
                       #mincnt=1,
    
    jg.ax_joint.set_xlabel(r'$\alpha_z$')
    jg.ax_joint.set_ylabel(r'$\alpha_m$')
    jg.ax_joint.plot([-5, 1.], [-5, 1.], 'k--')
    #jg.ax_joint.set_ylim(ymnx)

    plotting.set_fontsize(jg.ax_joint, 15.)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_2dstats(outroot='fig_2dstats_', stat=None,
                local=False, vmax=None, 
                cmap=None, cuts=None, scl = 1, debug=False):

    # Load table
    modis_tbl = load_modis_tbl(local=local, cuts=cuts)

    # Debug?
    if debug:
        modis_tbl = modis_tbl.loc[np.arange(1000000)].copy()

    # Stat
    if stat is None:
        stat = 'min_slope'
    lbls = dict(min_slope=r'$\alpha_{\rm min}$')
    if cmap is None:
        cmap = 'hot'
    outfile = outroot+stat+'.png'

    # Do it
    median_slope, x_edge, y_edge, ibins = scipy.stats.binned_statistic_2d(
        modis_tbl.U0, modis_tbl.U1, modis_tbl[stat],
        statistic='median', expand_binnumbers=True, bins=[24, 24])

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
    cbaxes.set_label(f'median({lbls[stat]})', fontsize=17.)
    cbaxes.ax.tick_params(labelsize=15)

    ax.set_xlabel(r'$U_0$')
    ax.set_ylabel(r'$U_1$')

    plotting.set_fontsize(ax, 17.)
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



        
#########################################################
### function used to create the learning plots

def fig_train_valid_learn_curve(opt_path: str):
    # loading parameters json file
    opt = Params(opt_path)
    opt = option_preprocess(opt)
   
    # build data loader
    train_loader = modis_loader_v2(opt)
    valid_loader = modis_loader_v2(opt, valid=True)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    # build loss list
    loss_train, loss_step_train, loss_avg_train = [], [], []
    loss_valid, loss_step_valid, loss_avg_valid = [], [], []
    # training routine
    for epoch in trange(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, losses_step, losses_avg = train_learn_curve(train_loader, model, criterion, optimizer, epoch, opt, cuda_use=opt.cuda_use)
        
        # record train loss
        loss_train.append(loss)
        loss_step_train += losses_step
        loss_avg_train += losses_avg
        
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        if epoch % opt.valid_freq == 0:
            epoch_valid = epoch // opt.valid_freq
            time1_valid = time.time()
            loss, losses_step, losses_avg = valid_learn_curve(valid_loader, model, criterion, epoch_valid, opt, cuda_use=opt.cuda_use)
           
            # record valid loss
            loss_valid.append(loss)
            loss_step_valid += losses_step
            loss_avg_valid += losses_avg
        
            time2_valid = time.time()
            print('valid epoch {}, total time {:.2f}'.format(epoch_valid, time2_valid - time1_valid))
            
    if not os.path.isdir('./learning_curve/'):
        os.mkdir('./learning_curve/')
        
    losses_file_train = f'./learning_curve/{opt.dataset}_losses_train.h5'
    losses_file_valid = f'./learning_curve/{opt.dataset}_losses_valid.h5'
    
    with h5py.File(losses_file_train, 'w') as f:
        f.create_dataset('loss_train', data=np.array(loss_train))
        f.create_dataset('loss_step_train', data=np.array(loss_step_train))
        f.create_dataset('loss_avg_train', data=np.array(loss_avg_train))
    with h5py.File(losses_file_valid, 'w') as f:
        f.create_dataset('loss_valid', data=p.array(loss_valid))
        f.create_dataset('loss_step_valid', data=np.array(loss_step_valid))
        f.create_dataset('loss_avg_valid', data=np.array(loss_avg_valid))
    
#### ########################## #########################
def main(pargs):

    # UMAP gallery
    if pargs.figure == 'augment':
        fig_augmenting()

    # UMAP LL
    if pargs.figure == 'umap_LL':
        # LL
        #fig_umap_colored(local=local)
        # DT
        fig_umap_colored(local=local, metric='DT', outfile='fig_umap_DT.png',
                         vmnx=(None, None))
        # Clouds
        #fig_umap_colored(local=local, metric='clouds', outfile='fig_umap_clouds.png',
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
                    point_size=1., 
                    lbl=r'Brazil, $\Delta T \approx 2$K',
                    vmnx=(-400, 400))

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

    # 2D Stats
    if pargs.figure == '2d_stats':
        fig_2dstats(local=pargs.local, debug=pargs.debug)
        
    # learning_curve
    if pargs.figure == 'learning_curve':
        opt_path = './experiments/opt.json'
        fig_train_valid_learn_curve(opt_path)

# Command line execution
if __name__ == '__main__':

    pargs = parse_option()
    main(pargs)
