""" Figures for the GHRSST 2021 Meeting"""
import os, sys
import numpy as np
import glob


import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

mpl.rcParams['font.family'] = 'stixgeneral'

import healpy as hp
import h5py

import torch

import pandas
import seaborn as sns

from ulmo import plotting
from ulmo import io as ulmo_io
from ulmo.utils import utils as utils
from ulmo.ssl import single_image as ssl_simage

from IPython import embed


def fig_augmenting(outfile='fig_augmenting.png', use_s3=False):


    # Load up an image
    if use_s3:
        modis_dataset_path = 's3://modis-l2/PreProc/MODIS_R2019_2003_95clear_128x128_preproc_std.h5'
    else:
        modis_dataset_path = "/home/xavier/Projects/Oceanography/AI/OOD/MODIS_L2/PreProc/MODIS_R2019_2003_95clear_128x128_preproc_std.h5"
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
                yticklabels=[], cmap=cm, cbar=True)
    
    # Augment me
    loader = ssl_simage.image_loader(img)
    test_batch = iter(loader).next()
    img1, img2 = test_batch

    ax1 = plt.subplot(gs[1])
    sns.heatmap(img1[0,0,...], ax=ax1, xticklabels=[], 
                yticklabels=[], cbar=False, cmap=cm)
    ax2 = plt.subplot(gs[2])
    sns.heatmap(img2[0,0,...], ax=ax2, xticklabels=[], 
                yticklabels=[], cbar=False, cmap=cm)

    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # Month histogram
    if flg_fig & (2 ** 0):
        fig_augmenting()

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        flg_fig += 2 ** 0  # Augmenting
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)

