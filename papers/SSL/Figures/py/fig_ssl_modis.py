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


# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import ssl_paper_analy

def update_outfile(outfile, table, umap_dim=2,
                   umap_comp=None):
    # Table
    if table is None or table == 'std':
        pass
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
                region=None,
                umap_comp='0,1',
                umap_dim=2,
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
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, cuts=cuts, 
                                              region=region, table=table,
                               percentiles=percentiles)
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
    elif metric == 'DT':
        values = np.log10(modis_tbl.DT.values)
        lmetric = r'$\log \, \Delta T$'
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
    img = ax0.scatter(modis_tbl[umap_keys[0]], modis_tbl[umap_keys[1]],
            s=point_size, c=values,
            cmap=cmap, vmin=vmnx[0], vmax=vmnx[1])
    cb = plt.colorbar(img, pad=0., fraction=0.030)
    cb.set_label(lmetric, fontsize=14.)
    #
    ax0.set_xlabel(r'$'+umap_keys[0]+'$')
    ax0.set_ylabel(r'$'+umap_keys[1]+'$')
    #ax0.set_aspect('equal')#, 'datalim')

    fsz = 17.
    plotting.set_fontsize(ax0, fsz)

    # Set boundaries
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
                     local=False, table='std', in_vmnx=None,
                     umap_comp='0,1',
                     umap_dim=2,
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
    fig = plt.figure(figsize=(8, 8))
    plt.clf()
    ax = plt.gca()

    ax.set_xlabel(r'$'+umap_keys[0]+'$')
    ax.set_ylabel(r'$'+umap_keys[1]+'$')

    # Gallery
    #dxdy=(0.3, 0.3)
    #xmin, xmax = modis_tbl.U0.min()-dxdy[0], modis_tbl.U0.max()+dxdy[0]
    #ymin, ymax = modis_tbl.U1.min()-dxdy[1], modis_tbl.U1.max()+dxdy[1]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

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
    for x in xval[:-1]:
        for y in yval[:-1]:
            pts = np.where((modis_tbl[umap_keys[0]] >= x) & (
                modis_tbl[umap_keys[0]] < x+dxv) & (
                modis_tbl[umap_keys[1]] >= y) & (modis_tbl[umap_keys[1]] < y+dxv)
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
            _ = sns.heatmap(np.flipud(cutout_img), 
                            xticklabels=[], 
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
                    table=None,
                    version=1, local=False, vmax=None, 
                    cmap=None, cuts=None, region=None,
                    scl = 1):

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
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, cuts=cuts)

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
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, cuts=cuts)

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
                   local=False, vmax=None, 
                   distr='normal',
                   cmap=None, cuts=None, debug=False):

    # Load table
    modis_tbl = ssl_paper_analy.load_modis_tbl(local=local, cuts=cuts)

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
    if pargs.figure == 'umap_DT':
        fig_umap_colored(local=pargs.local, table=pargs.table,
                         metric='DT', outfile='fig_umap_DT.png',
                         vmnx=(None,None),
                         umap_dim=pargs.umap_dim,
                         umap_comp=pargs.umap_comp)
        #                 vmnx=(None, None), table=pargs.table,
        #                 umap_dim=pargs.umap_dim,
        #                 umap_comp=pargs.umap_comp)
        # Clouds
        #fig_umap_colored(local=pargs.local, metric='clouds', outfile='fig_umap_clouds.png',
        #                 vmnx=(None,None))

    # UMAP gallery
    if pargs.figure == 'umap_gallery':
        #fig_umap_gallery(debug=pargs.debug, in_vmnx=(-5.,5.), table=pargs.table) 
        #fig_umap_gallery(debug=pargs.debug, in_vmnx=None, table=pargs.table,
        #                 outfile='fig_umap_gallery_novmnx.png')
        if pargs.vmnx is not None:
            vmnx = [int(ivmnx) for ivmnx in pargs.vmnx.split(',')]
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
        fig_slopes(local=pargs.local, debug=pargs.debug)

    # Slope vs DT
    if pargs.figure == 'slopevsDT':
        fig_slopevsDT(local=pargs.local, debug=pargs.debug)
    
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
