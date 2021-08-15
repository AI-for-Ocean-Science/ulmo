""" Figures for SSL paper on MODIS """
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

local_modis_file = os.path.join(os.getenv('SST_OOD'),
                                'MODIS_L2/Tables/MODIS_L2_std.parquet')

def load_modis_tbl(tbl_file=None, local=False, cuts=None):
    if tbl_file is None:
        tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    if local:
        tbl_file = local_modis_file

    # Load
    modis_tbl = ulmo_io.load_main_table(tbl_file)
    if 'DT' not in modis_tbl.keys():
        modis_tbl['DT'] = modis_tbl.T90 - modis_tbl.T10

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
                    cmap=None, cuts=None, scl = 1):

    # Load table
    modis_tbl = load_modis_tbl(local=local, cuts=cuts)

    # Plot
    fig = plt.figure(figsize=(12, 12))
    plt.clf()


    jg = sns.jointplot(data=modis_tbl, x='DT', y='LL',
                  kind='hex')

    #ax.set_xlabel(r'$U_0$')
    #ax.set_ylabel(r'$U_1$')

    plotting.set_fontsize(jg.ax_joint, 15.)
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
def main(flg_fig, local, debug):

    # UMAP gallery
    if flg_fig == 'augment':
        fig_augmenting()

    # UMAP LL
    if flg_fig == 'umap_LL':
        # LL
        #fig_umap_colored(local=local)
        # DT
        fig_umap_colored(local=local, metric='DT', outfile='fig_umap_DT.png',
                         vmnx=(None, None))
        # Clouds
        #fig_umap_colored(local=local, metric='clouds', outfile='fig_umap_clouds.png',
        #                 vmnx=(None,None))

    # UMAP gallery
    if flg_fig == 'umap_gallery':
        fig_umap_gallery(debug=debug, in_vmnx=(-5.,5.)) 
        fig_umap_gallery(debug=debug, in_vmnx=None,
                         outfile='fig_umap_gallery_novmnx.png')
        fig_umap_gallery(debug=debug, in_vmnx=(-1.,1.), 
                         outfile='fig_umap_gallery_vmnx1.png')

    # UMAP LL Brazil
    if flg_fig  == 'umap_brazil':
        fig_umap_colored(outfile='fig_umap_brazil.png', 
                    region='brazil',
                    point_size=1., 
                    lbl=r'Brazil, $\Delta T \approx 2$K',
                    vmnx=(-400, 400))

    # UMAP 2d Histogram
    if flg_fig == 'umap_2dhist':
        #
        fig_umap_2dhist(vmax=80000, local=local)
        # Near norm
        fig_umap_2dhist(outfile='fig_umap_2dhist_inliers.png',
                        local=local, cmap='Greens', 
                        cuts='inliers')

    # LL vs DT
    if flg_fig == 'LLvsDT':
        fig_LLvsDT(local=local, debug=debug)


# Command line execution
if __name__ == '__main__':

    local = True if 'local' in sys.argv else False
    debug = True if 'debug' in sys.argv else False
    if len(sys.argv) == 1:
        flg_fig = 'LLvsDT'
        #flg_fig += 2 ** 0  # Augmenting
        #flg_fig += 2 ** 1  # UMAP colored by various things
        #flg_fig += 2 ** 2  # UMAP SSL gallery
        #flg_fig += 2 ** 3  # UMAP brazil
        #flg_fig += 2 ** 4  # UMAP 2d Histogram
        #flg_fig += 2 ** 5  # 
    else:
        flg_fig = sys.argv[1]

    main(flg_fig, local, debug)

