""" Plotting routines for the MAE """

import numpy as np

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import seaborn as sns

from ulmo import plotting
from ulmo.mae import patch_analysis

def plot_recon(orig_img, recon_img, mask_img, p_sz:int=4,
               outfile:str=None, gs:gridspec=None,
               bias:float=0., 
               img_vmnx:tuple=(-1,1),
               res_vmnx:tuple=(None,None),
               show_title:bool=True):

    # Prep
    diff_true = recon_img - orig_img 
    patches = patch_analysis.find_patches(mask_img, p_sz)

    if gs is None:
        fig = plt.figure(figsize=(7, 3.1))
        plt.clf()
        gs = gridspec.GridSpec(1,3)
    ax0 = plt.subplot(gs[0])

    _, cm = plotting.load_palette()
    cbar_kws={'label': 'SSTa (K)', 
              'fraction': 0.0450,
              'location': 'top'}
    _ = sns.heatmap(np.flipud(orig_img), xticklabels=[], 
                     vmin=img_vmnx[0], vmax=img_vmnx[1],
                     yticklabels=[], cmap=cm, cbar=True, 
                     square=True, 
                     cbar_kws=cbar_kws,
                     ax=ax0)

    # Reconstructed
    sub_recon = np.ones_like(recon_img) * np.nan
    # Difference
    diff = np.ones_like(recon_img) * np.nan
    frecon = recon_img.copy()

    # Plot/fill the patches
    for kk, patch in enumerate(patches):
        i, j = np.unravel_index(patch, mask_img.shape)
        #print(i,j)
        rect = Rectangle((j,60-i), p_sz, p_sz,
            linewidth=0.5, edgecolor='k', 
            facecolor='none', ls='-', zorder=10)
        ax0.add_patch(rect)
        # Fill
        sub_recon[i:i+p_sz, j:j+p_sz] = recon_img[i:i+p_sz, j:j+p_sz] - bias
        frecon[i:i+p_sz, j:j+p_sz] -= bias
        # TODO -- Turn this off
        diff[i:i+p_sz, j:j+p_sz] = diff_true[i:i+p_sz, j:j+p_sz] - bias

    # Recon image
    ax1 = plt.subplot(gs[1])

    '''
    if full_recon:
        sub_recon = frecon.copy()
    '''
    _ = sns.heatmap(np.flipud(sub_recon), xticklabels=[], 
                     vmin=img_vmnx[0], vmax=img_vmnx[1],
                     yticklabels=[], cmap=cm, cbar=True, 
                     square=True, cbar_kws=cbar_kws,
                     ax=ax1)
    

    # Recon image
    ax2 = plt.subplot(gs[2])

    cbar_kws['label'] = 'Residuals (K)'
    _ = sns.heatmap(np.flipud(diff), xticklabels=[], 
                     vmin=res_vmnx[0], vmax=res_vmnx[1],
                     yticklabels=[], cmap='bwr', cbar=True, 
                     square=True, 
                     cbar_kws=cbar_kws,
                     ax=ax2)

    # Borders
    # 
    for ax, title in zip( [ax0, ax1, ax2],
        ['Original', 'Reconstructed', 'Residuals']):
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth(1.)  
        #
        if show_title:
            ax.set_title(title, fontsize=14, y=-0.13)


    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        plt.close()
        print('Wrote {:s}'.format(outfile))