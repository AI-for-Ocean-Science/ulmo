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

from IPython import embed

local_modis_file = os.path.join(os.getenv('SST_OOD'),
                                'MODIS_L2/Tables/MODIS_L2_std.parquet')


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

def fig_umap_gallery(outfile='fig_umap_gallery.png',
                     version=1, local=True, restrict_DT=False,
                     debug=False): 
    if version == 1:                    
        tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    else:
        raise IOError("bad version number")
    if local:
        tbl_file = local_modis_file
    # Load
    modis_tbl = ulmo_io.load_main_table(tbl_file)
    num_samples = len(modis_tbl)

    if debug: # take a subset
        print("DEBUGGING IS ON")
        nsub = 500000
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        idx = idx[0:nsub]
        modis_tbl = modis_tbl.loc[idx].copy()


    # Restrict on DT?
    #if restrict_DT:
    #    llc_tbl['DT'] = llc_tbl.T90 - llc_tbl.T10
    #    llc_tbl = llc_tbl[ll]

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1, 2)

    # Just the UMAP colored by LL
    ax0 = plt.subplot(gs[0])

    point_size = 1. / np.sqrt(num_samples)
    img = ax0.scatter(modis_tbl.U0, modis_tbl.U1,
            s=point_size, c=modis_tbl.LL, 
            cmap='jet', vmin=-1000., vmax=None)
    cb = plt.colorbar(img, pad=0.)
    cb.set_label('LL', fontsize=20.)
    #
    ax0.set_xlabel(r'$U_0$')
    ax0.set_ylabel(r'$U_1$')
    ax0.set_aspect('equal', 'datalim')

    fsz = 15.
    set_fontsize(ax0, fsz)

    # Set boundaries
    dxdy=(0.3, 0.3)
    xmin, xmax = modis_tbl.U0.min()-dxdy[0], modis_tbl.U0.max()+dxdy[0]
    ymin, ymax = modis_tbl.U1.min()-dxdy[1], modis_tbl.U1.max()+dxdy[1]
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)

    # Gallery
    ax1 = plt.subplot(gs[1])
    _ = plotting.umap_gallery(modis_tbl, ax=ax1, skip_scatter=True, dxdy=dxdy,
                              fsz=fsz)

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

    # UMAP gallery
    if flg_fig & (2 ** 0):
        fig_augmenting()

    # UMAP gallery
    if flg_fig & (2 ** 1):
        fig_umap_gallery(debug=True)

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # Augmenting
        flg_fig += 2 ** 1  # UMAP SSL gallery
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)

