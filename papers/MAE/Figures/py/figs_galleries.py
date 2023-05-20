""" Create SST cutout galleries """
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

import pandas as pd
import seaborn as sns

import h5py

from ulmo import plotting
from ulmo.utils import utils as utils

from ulmo.mae import patch_analysis


from IPython import embed

def patches(orig_img, mask_img, p_sz):
    # Find the patches
    p_sz = 4
    unmasked = 1 - mask_img
    patches = patch_analysis.find_patches(mask_img, p_sz)
    upatches = patch_analysis.find_patches(unmasked, p_sz)
    
    # Reconstructed
    sub_recon = np.ones_like(orig_img) * np.nan
    
    # Plot/fill the patches
    for kk, patch in enumerate(upatches):
        i, j = np.unravel_index(patch, unmasked.shape)
        # Fill
        sub_recon[i:i+p_sz, j:j+p_sz] = orig_img[i:i+p_sz, j:j+p_sz]
    return sub_recon
    

def plot_gallery(imgs_file, outfile='mask_ratio_gallery.png', vmnx = [None, None]):
    """
    Gallery to show how different models perform at different masking ratios
    default file to run this is 'idx330469_recons.npz' uploaded in the Enki google doc
    imgs_file: npz file containing 
               1) 'img' containing the original image
               2) 'recons' with p=10,30,50 reconstructions 
                   for t=10,35,50,75 in that order (i.e. imgs[1][2] is p=30, t=35 recons)
               3) 'masks' with the mask for p=10,30,50 in that order
    outfile:   Name of output file
    vmnx:      The lower and upper limits for the colorbar
    """
    imgs = np.load(imgs_file)
    orig_img = imgs['img']
    p = imgs['recons']
    masks = imgs['masks']
    print(np.max(orig_img))
    print(np.min(orig_img))
    
    fig = plt.figure(figsize=(12, 7))
    #fig.tight_layout()
    plt.clf()
    gs = gridspec.GridSpec(3,6)
    ax0 = plt.subplot(gs[1:2, 0])
    
    _, cm = plotting.load_palette()
    cbar_kws={'label': 'SSTa (K)',
              'fraction': 0.0450,
              'location': 'top'}
    
    # Plot original image
    _ = sns.heatmap(np.flipud(orig_img), xticklabels=[],
                    vmin=vmnx[0], vmax=vmnx[1],
                    yticklabels=[], cmap=cm, cbar=True, 
                    square=True, 
                    cbar_kws=cbar_kws,
                    ax=ax0)
    
    # Plot Masks
    ax1 = plt.subplot(gs[0, 1])
    sub_recon = patches(orig_img, masks[0], 4)
    _ = sns.heatmap(np.flipud(sub_recon), xticklabels=[],
                vmin=vmnx[0], vmax=vmnx[1],
                yticklabels=[], cmap=cm, cbar=False, 
                square=True, ax=ax1)
    ax2 = plt.subplot(gs[1, 1])
    sub_recon = patches(orig_img, masks[1], 4)
    _ = sns.heatmap(np.flipud(sub_recon), xticklabels=[],
                vmin=vmnx[0], vmax=vmnx[1],
                yticklabels=[], cmap=cm, cbar=False, 
                square=True, ax=ax2)
    ax3 = plt.subplot(gs[2, 1])
    sub_recon = patches(orig_img, masks[2], 4)
    _ = sns.heatmap(np.flipud(sub_recon), xticklabels=[],
                vmin=vmnx[0], vmax=vmnx[1],
                yticklabels=[], cmap=cm, cbar=False, 
                square=True, ax=ax3)
    
    # Plot reconstructions
    model = [10,35,50,75]
    for i in range(3):
        for j in range(4):
            ax = plt.subplot(gs[i, j+2])
            ax.patch.set_edgecolor('black')  
            ax.patch.set_linewidth(1.)
            ax.text(2,6, 't{}'.format(model[j]), c='w', weight='bold', fontsize='12')
            #ax.text(2,6, 't{}'.format(model[j]), c='k', weight='bold', fontsize='12')
            _ = sns.heatmap(np.flipud(p[i][j]), xticklabels=[],
                    vmin=vmnx[0], vmax=vmnx[1],
                    yticklabels=[], cmap=cm, cbar=False, 
                    square=True, ax=ax)
    fig.subplots_adjust(wspace=0.1, hspace=0)
    
     # Borders
    for ax, title in zip( [ax0, ax1, ax2 ,ax3],
        ['original', 'p = 10', 'p = 30', 'p = 50']):
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth(1.)  
        
        show_title=True
        if show_title:
            ax.set_title(title, fontsize=14, y=-0.18)
    
    # Adjust whitespace and plot position
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.05)
    
    # TODO: move colorbar to bottom or side of plot
    
    # Save
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f'Wrote: {outfile}')
    return
    
    

def create_gallery(filepath='data/gallery_imgs.npz', outfile='recon_gallery.png', vmnx = [-2, 2]):
    """
    Creates a gallery of images with different LL for the different models
    gallery_imgs.npz is a file containing orig_imgs, recon_imgs, and mask_imgs np arrays
    The indexing is consistent across all arrays, and is ordered so the first four images
    are for LL=3 reconstructions, the next 4 are LL=126, etc.
    """
    
    # Load images
    imgs = np.load(filepath)
    orig_imgs = imgs['orig_imgs']
    recon_imgs = imgs['recon_imgs']
    masks = imgs['masks']

    # Create Plot
    fig = plt.figure(figsize=(12, 16))
    #outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    outer = gridspec.GridSpec(9, 7, wspace=0.2, hspace=0.2)
    _, cm = plotting.load_palette()

    # Loop through subplots (sorted by LL)
    for i in range(4):
        #inner = gridspec.GridSpecFromSubplotSpec(4, 3,
        #                subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        # For every row, post orig_img, mask_img, recon_img
        # Loop through models of p and plot orig_img, mask_img, recon_imgs
        if i in [0,1]:
            col0 = 0
        else:
            col0 = 4
        if i in [0,2]:
            row0 = 0
        else:
            row0 = 5

        for j in range(4):
            orig_img = orig_imgs[i*4+j]
            recon_img = recon_imgs[i*4+j]
            mask_img = masks[i*4+j]



            # orig_img
            """
            cbar_kws={'label': 'SSTa (K)',
            'fraction': 0.0450,
            'location': 'top'}
            """
            
            #ax0 = plt.Subplot(fig, inner[j*3])
            ax0 = plt.subplot(outer[row0+j, col0])
            _ = sns.heatmap(np.flipud(orig_img), xticklabels=[],
                            vmin=vmnx[0], vmax=vmnx[1],
                            yticklabels=[], cmap=cm, cbar=False,
                            square=True, 
                            ax=ax0)
            fig.add_subplot(ax0)

            # mask_img (use patches)
            sub_recon = patches(orig_img, mask_img, 4)
            ax1 = plt.subplot(outer[row0+j, col0+1])
            #ax1 = plt.Subplot(fig, inner[j*3+1])
            _ = sns.heatmap(np.flipud(sub_recon), xticklabels=[],
                            vmin=vmnx[0], vmax=vmnx[1],
                            yticklabels=[], cmap=cm, cbar=False,  
                            square=True,
                            ax=ax1)
            fig.add_subplot(ax1)

            # recon_img
            #ax2 = plt.Subplot(fig, inner[j*3+2])
            ax2 = plt.subplot(outer[row0+j, col0+2])
            _ = sns.heatmap(np.flipud(recon_img), xticklabels=[],
                            vmin=vmnx[0], vmax=vmnx[1],
                            yticklabels=[], cmap=cm, cbar=False, 
                            square=True,
                            ax=ax2)
            fig.add_subplot(ax2)

            for ax in [ax0, ax1, ax2]:
                ax.patch.set_edgecolor('black')  
                ax.patch.set_linewidth(1.)

    # Color bars
    ax_cb1 = plt.subplot(outer[4, 0:3])
    img = ax_cb1.imshow(np.array([[0,1]]), cmap=cm, vmin=vmnx[0], vmax=vmnx[1])
    img.set_visible(False)
    plt.colorbar(img, orientation="horizontal", cax=ax_cb1)

    ax_cb2 = plt.subplot(outer[4, 4:7])
    img = ax_cb2.imshow(np.array([[0,1]]), cmap=cm, vmin=vmnx[0], vmax=vmnx[1])
    img.set_visible(False)
    plt.colorbar(img, orientation="horizontal", cax=ax_cb2)

    # Finish
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))
    return

create_gallery()