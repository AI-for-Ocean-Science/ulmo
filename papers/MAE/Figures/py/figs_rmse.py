import sys
import os
import requests
import time

import torch
import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from ulmo.plotting import plotting
from ulmo import io as ulmo_io

from IPython import embed
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

def fig_batch_rmse(model, rmse_filepath='valid_avg_rms.csv'):
    """
    Creates a figure of average RMSE by LL batches for a single image.
    model: MAE model (10, 35, 50, 75)
    rmse_filepath: file with rmse's
    """
    # load rmse
    rmse = pd.read_csv(rmse_filepath)
    
    # Plot
    fig, ax = plt.subplots()
    
    masks = [10, 20, 30, 40, 50]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    plt_labels = []
    for p, c in zip(masks, colors):
        x = rmse['median_LL']
        y = rmse['rms_t{t}_p{p}'.format(t=model, p=p)]
        plt_labels.append('p={p}%'.format(p=p))
        plt.scatter(x, y, color=c)

    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
    plt.legend(labels=plt_labels, title='Masking Ratio',
               title_fontsize='small', fontsize='small', fancybox=True)
    plt.title('Average RMSE vs LL: t={}'.format(model))
    plt.xlabel("Median LL Per Batch")
    plt.ylabel("RMSE")

    # save
    outfile = 'rmse_t{}.png'.format(model)
    plt.savefig(outfile, dpi=300)
    plt.close()

    return

def figs_rmse_all_models(outfile='rmse_models.png',
                         rmse_filepath='../Analysis/valid_avg_rms.csv'):
    # load rmse
    rmse = pd.read_csv(rmse_filepath)
    
    fig = plt.figure(figsize=(10, 10))
    
    plt.clf()
    gs = gridspec.GridSpec(2,2)
    
    models = [10,35,50,75]
    masks = [10,20,30,40,50]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    plt_labels = []
    for i in range(4):
        
        # determine position of plot
        pos = [0, i]
        if i > 1:
            pos = [1, i-2]
        ax = plt.subplot(gs[pos[0], pos[1]])
        
        # plot
        for p, c in zip(masks, colors):
            x = rmse['median_LL']
            y = rmse['rms_t{t}_p{p}'.format(t=models[i], p=p)]
            plt_labels.append('p={p}%'.format(p=p))
            plt.scatter(x, y, color=c)

        if models[i] != 50:
            ax.set_ylim([0, 0.15])
        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
        plt.legend(labels=plt_labels, title='Masking Ratio',
                   title_fontsize='small', fontsize='small', fancybox=True)
        plt.title('t={}'.format(models[i]))
        plt.xlabel("Median LL Per Batch")
        plt.ylabel("Average RMSE")
                         
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.subplots_adjust(wspace=0.2)
    fig.suptitle('RMSE vs LL', fontsize=16)
    
    outfile = 'rmse_vs_LL.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f'Wrote: {outfile}')
    return

def figs_rmse_t10(outfile='rmse_t10only.png',
                         rmse_filepath='../Analysis/valid_avg_rms.csv'):
    # load rmse
    rmse = pd.read_csv(rmse_filepath)
    
    fig = plt.figure(figsize=(10, 10))
    
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    
    models = [10,35,50,75]
    masks = [10,20,30,40,50]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    plt_labels = []
    for i in range(1):
        
        # determine position of plot
        ax = plt.subplot(gs[0])
        
        # plot
        for p, c in zip(masks, colors):
            x = rmse['median_LL']
            y = rmse['rms_t{t}_p{p}'.format(t=models[i], p=p)]
            plt_labels.append('p={p}'.format(p=p))
            plt.scatter(x, y, color=c)

        ax.set_ylim([0, 0.10])
        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
        plt.legend(labels=plt_labels, title='Training Ratio',
                   title_fontsize='large', fontsize='large', fancybox=True)
        plt.title('t={}'.format(models[i]))
        plt.xlabel("Median LL Per Batch")
        plt.ylabel("Average RMSE")

        plotting.set_fontsize(ax, 19)
                         
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.subplots_adjust(wspace=0.2)
    #fig.suptitle('RMSE vs LL', fontsize=16)
    
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f'Wrote: {outfile}')
    return
    
#figs_rmse_all_models()
figs_rmse_t10()

'''
models = [10,35,50,75]

for t in models:
    fig_batch_rmse(t)
'''