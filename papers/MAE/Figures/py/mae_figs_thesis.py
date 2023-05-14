from dataclasses import replace
from datetime import datetime
import os, sys, requests
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
from scipy.optimize import curve_fit

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas as pd
import seaborn as sns
from matplotlib import ticker
import h5py

from ulmo import plotting
from ulmo.utils import utils as utils
from ulmo import io as ulmo_io
from ulmo.ssl import single_image as ssl_simage
from ulmo.ssl import defs as ssl_defs
from ulmo.mae import patch_analysis
from ulmo.mae import models_mae
from ulmo.utils import image_utils


import requests

import torch

from PIL import Image
from ulmo.plotting import plotting



##############################################################
# ------------- Generate Cloud Coverage Plot------------------
##############################################################
def fig_cloud_coverage(filepath='data/modis_2020_cloudcover.npz', 
                       outfile='cloud_coverage.png'):

    #filepath = 'data/modis_2020_cloudcover.npz'

    data = np.load(filepath)
    lst = data.files
    x = data['CC_values']
    y = data['tot_pix_CC']

    from scipy.interpolate import make_interp_spline, BSpline
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(np.min(x), np.max(x), 300)

    spl = make_interp_spline(x, y, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    sns.set_style("whitegrid")
    sns.set_context("paper")

    f, ax = plt.subplots(figsize=(8, 7))
    #ax.set_axisbelow(True)
    #ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)

    sns.lineplot(x=xnew, y=power_smooth, color='blue', linewidth=2.5)
    
    ax.set_yscale("log")
    #plt.plot(xnew, power_smooth)
    ax.set_xlim(0,1)
    ax.set_ylim(10**7,10**11)
    #ax.xaxis.set_ticks(np.arange(0, 1, 0.1))
    ax.set_xlabel('Percentage of Clouds in Cutout (CC)')
    ax.set_ylabel(f'Cutouts Available ($N_c$)')
    #ax.set_title("Cutouts vs Cloud Coverage")

    #sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
    #ax.tick_params(which="both", bottom=True)
    ax.tick_params(axis='y', which='both', direction='out', length=4, left=True,
                   color='gray')
    ax.grid(True, which='both', color='gray', linewidth=0.1)
    ax.minorticks_on()



    plotting.set_fontsize(ax, 15)

    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f'Wrote: {outfile}')


##############################################################
# --------------- Generate Training Image --------------------
##############################################################
"""
For single image reconstructions.
"""
def prepare_model(chkpt_dir, arch='mae_vit_LLC_patch4'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, mask_ratio):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *1)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    
    im = im_paste.cpu().detach().numpy()
    m = mask.cpu().detach().numpy()
    re = y.cpu().detach().numpy()
    im = im.squeeze()
    m = m.squeeze()
    re = re.squeeze()
    print('reconstruction complete')
    
    return im, m, re

def plot_recon(orig_img, recon_img, recon_full, mask_img, idx,
               apply_bias=False, vmnx = [None, None],
               LL_file='MAE_LLC_valid_nonoise.parquet'):
    # Load Unmasked
    unmasked = 1 - mask_img

    # Bias
    diff_true = recon_img - orig_img
    bias = np.median(diff_true[np.abs(diff_true)>0.])

    # Find the patches
    p_sz = 4
    patches = patch_analysis.find_patches(mask_img, p_sz)
    upatches = patch_analysis.find_patches(unmasked, p_sz)


    fig = plt.figure(figsize=(13, 4))
    plt.clf()
    gs = gridspec.GridSpec(1,5)
    ax0 = plt.subplot(gs[0])

    _, cm = plotting.load_palette()
    cbar_kws={'label': 'SSTa (K)',
              'fraction': 0.0450,
              'location': 'top'}

    _ = sns.heatmap(np.flipud(orig_img), xticklabels=[],
                    vmin=vmnx[0], vmax=vmnx[1],
                    yticklabels=[], cmap=cm, cbar=True, 
                    square=True, 
                    cbar_kws=cbar_kws,
                    ax=ax0)

    # Reconstructed
    sub_recon = np.ones_like(recon_img) * np.nan
    # Difference
    diff = np.ones_like(recon_img) * np.nan
    frecon = recon_img.copy()

    # Reconstructed
    usub_recon = np.ones_like(recon_img) * np.nan
    # Difference
    udiff = np.ones_like(recon_img) * np.nan
    ufrecon = recon_img.copy()

    # Plot/fill the patches for masked image
    for kk, patch in enumerate(upatches):
        i, j = np.unravel_index(patch, unmasked.shape)
        # Fill
        usub_recon[i:i+p_sz, j:j+p_sz] = orig_img[i:i+p_sz, j:j+p_sz]
        ufrecon[i:i+p_sz, j:j+p_sz]
        # ???
        udiff[i:i+p_sz, j:j+p_sz] = diff_true[i:i+p_sz, j:j+p_sz]

    # Unmasked image
    ax1 = plt.subplot(gs[1])

    u_recon = False
    if u_recon:
        usub_recon = ufrecon.copy()
    _ = sns.heatmap(np.flipud(usub_recon), xticklabels=[],
                    vmin=vmnx[0], vmax=vmnx[1],
                    yticklabels=[], cmap=cm, cbar=True, 
                    square=True, cbar_kws=cbar_kws,
                    ax=ax1)

    
    # Plot/fill the patches for latent vector
    for kk, patch in enumerate(upatches):
        i, j = np.unravel_index(patch, unmasked.shape)
        # Fill
        usub_recon[i:i+p_sz, j:j+p_sz] = recon_full[i:i+p_sz, j:j+p_sz]
        ufrecon[i:i+p_sz, j:j+p_sz]
        # ???
        udiff[i:i+p_sz, j:j+p_sz] = diff_true[i:i+p_sz, j:j+p_sz]
    
    # Unmasked image
    ax2 = plt.subplot(gs[2])

    u_recon = False
    if u_recon:
        usub_recon = ufrecon.copy()
    _ = sns.heatmap(np.flipud(usub_recon), xticklabels=[],
                    vmin=vmnx[0], vmax=vmnx[1],
                    yticklabels=[], cmap=cm, cbar=True, 
                    square=True, cbar_kws=cbar_kws,
                    ax=ax2)
    
    # Full Recon image
    ax3 = plt.subplot(gs[3])

    full_recon = True
    if apply_bias:
        cbar_kws['label'] = 'SSTa (K) ({:.3f} bias)'.format(bias)
    if full_recon:
        sub_recon = frecon.copy()
    _ = sns.heatmap(np.flipud(recon_full), xticklabels=[],
                    vmin=vmnx[0], vmax=vmnx[1],
                    yticklabels=[], cmap=cm, cbar=True, 
                    square=True, cbar_kws=cbar_kws,
                    ax=ax3)

    # Recon image
    ax4 = plt.subplot(gs[4])

    full_recon = True
    if apply_bias:
        cbar_kws['label'] = 'SSTa (K) ({:.3f} bias)'.format(bias)
    if full_recon:
        sub_recon = frecon.copy()
    _ = sns.heatmap(np.flipud(sub_recon), xticklabels=[],
                    vmin=vmnx[0], vmax=vmnx[1],
                    yticklabels=[], cmap=cm, cbar=True, 
                    square=True, cbar_kws=cbar_kws,
                    ax=ax4)

    # Borders
    # 
    for ax, title in zip( [ax0, ax1, ax2 ,ax3, ax4],
        ['Original', 'Masked', 'Latent Representation', 'Decoder Results', 'Original + Reconstructed']):
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth(1.)  
        #
        show_title=True
        if show_title:
            ax.set_title(title, fontsize=14, y=-0.14)
    
    # Plot title
    table = pd.read_parquet(LL_file, engine='pyarrow',columns=['pp_idx', 'LL'])
    table = table[table['LL'].notna()]
    table = table.sort_values(by=['pp_idx'])
    LL = int(table.iloc[idx]['LL'])
    #fig.suptitle('{LL} LL Reconstruction: t{model} {p}% masking'.format(LL=LL))
    fig.suptitle('{LL} LL Reconstruction'.format(LL=LL))
    
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    outfile = 'training_visual.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def figs_training(idx=85674, 
                  filepath=os.path.join(os.getenv('OS_OGCM'),
                                              'LLC', 'Enki', 'PreProc', 
                                              'MAE_LLC_valid_nonoise_preproc.h5'), 
                  model_filepath=os.path.join(os.getenv('OS_OGCM'),
                                              'LLC', 'Enki', 'Models',
                                              'Enki_t75.pth'),
                  table = 'data/MAE_LLC_valid_nonoise.parquet'):
    """
    Create fig
    """
    # laod image and model
    f = h5py.File(filepath, 'r')
    img = f['valid'][idx][0]
    img.resize((64,64,1))
    model = prepare_model(model_filepath, 'mae_vit_LLC_patch4')
    print('Model75 loaded.')
    
    # Reconstruct
    recon_img, mask, full_recon = run_one_image(img, model, 0.75)
    orig_img = img.squeeze()
    
    plot_recon(orig_img, recon_img, full_recon, mask, idx, apply_bias=False, vmnx = [-1.8, 1.8], 
               LL_file=table)

# Command line execution
if __name__ == '__main__':

    figs_training()
    #fig_cloud_coverage()