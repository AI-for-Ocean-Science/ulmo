""" Figures for Talks related to the MAE paper """
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
from matplotlib.patches import Rectangle, Ellipse


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import h5py

from ulmo import plotting
from ulmo.utils import utils as utils

from ulmo import io as ulmo_io
from ulmo.ssl import single_image as ssl_simage
from ulmo.ssl import ssl_umap
from ulmo.ssl import defs as ssl_defs
from ulmo.mae import patch_analysis
from ulmo.utils import image_utils

from IPython import embed

# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import ssl_paper_analy
#sys.path.append(os.path.abspath("../Figures/py"))
#import fig_ssl_modis

def fig_recon_slide(outfile:str):

    img_path = '/home/xavier/Projects/Oceanography/data/MAE/Recon'
    orig_file = os.path.join(img_path, 'mae_reconstruct_t10_p20.h5')
    recon_file = os.path.join(img_path, 'mae_reconstruct_t10_p20.h5')
    mask_file = os.path.join(img_path, 'mae_mask_t10_p20.h5')

    # Load up images
    f_orig = h5py.File(orig_file, 'r')
    f_recon = h5py.File(recon_file, 'r')
    f_mask = h5py.File(mask_file, 'r')

    idx = 0
    orig_img = f_orig['valid'][idx,0,...]
    recon_img = f_recon['valid'][idx,0,...]
    mask_img = f_mask['valid'][idx,0,...]

    # Find the patches
    p_sz = 4
    patches = patch_analysis.find_patches(mask_img, p_sz)

    vmnx = -1, 1 #orig_img.min(), orig_img.max()
    
    fig = plt.figure(figsize=(7, 2))
    plt.clf()
    gs = gridspec.GridSpec(1,3)
    ax0 = plt.subplot(gs[0])

    _, cm = plotting.load_palette()
    cbar_kws={'label': 'SSTa (K)', 
              'fraction': 0.0450}
    _ = sns.heatmap(np.flipud(orig_img), xticklabels=[], 
                     vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap=cm, cbar=True, 
                     square=True, 
                     cbar_kws=cbar_kws,
                     ax=ax0)

    # Reconstructed
    sub_recon = np.ones_like(recon_img) * np.nan
    # Difference
    diff = recon_img - orig_img

    # Plot/fill the patches
    for kk, patch in enumerate(patches):
        i, j = np.unravel_index(patch, mask_img.shape)
        #print(i,j)
        rect = Rectangle((j,60-i), p_sz, p_sz,
            linewidth=0.5, edgecolor='k', 
            facecolor='none', ls='-', zorder=10)
        ax0.add_patch(rect)
        # Fill
        sub_recon[i:i+p_sz, j:j+p_sz] = recon_img[i:i+p_sz, j:j+p_sz]
        # TODO -- Turn this off
        diff[i:i+p_sz, j:j+p_sz] = np.random.uniform(size=(4,4), low=-0.1, high=0.1)

    # Recon image
    ax1 = plt.subplot(gs[1])

    _ = sns.heatmap(np.flipud(sub_recon), xticklabels=[], 
                     vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap=cm, cbar=True, 
                     square=True, cbar_kws=cbar_kws,
                     ax=ax1)

    # Recon image
    ax2 = plt.subplot(gs[2])

    _ = sns.heatmap(np.flipud(diff), xticklabels=[], 
                     #vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap='bwr', cbar=True, 
                     square=True, 
                     cbar_kws={'label': r'Residuals (K)',
                               'fraction': 0.0450},
                     ax=ax2)


    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # Uniform, non UMAP gallery for DTall
    if flg_fig & (2 ** 0):
        fig_recon_slide('fig_recon_slide.png')


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        flg_fig += 2 ** 0  # Reconstruction example
        #flg_fig += 2 ** 1  # Number satisfying
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)