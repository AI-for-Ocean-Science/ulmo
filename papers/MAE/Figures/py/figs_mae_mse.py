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

from ulmo.plotting import plotting

def fig_batch_mse(outfile: str,
                   model: str, labels, colors,
                    mse_filepath='valid_avg_mse.parquet'):
    """
    Creates a figure of average MSE by LL batches.
    outfile: file to save as
    model: MAE model (t10, t35, t75)
    labels: labels from pandas frame to plot
    mse_filepath: file with mses
    """
    # load mse
    mse = pd.read_parquet(mse_filepath, engine='pyarrow')
    
    fig, ax = plt.subplots()
    plt_labels = []    
    for l, c in zip(labels, colors):
        x = mse['avg_LL']
        y = mse[l]
        plt_labels.append('{}%'.format(l[-2:]))
        plt.scatter(x, y, color=c)
    
    plt.yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
    plt.legend(labels=plt_labels, title='Masking Ratio',
               title_fontsize='small', fontsize='small', fancybox=True, ncol=2)
    plt.title('{} Model: Avg MSE Based on Complexity'.format(model))
    plt.xlabel("Average LL Per Batch")
    plt.ylabel("log$_{10}$ MSE")

    # save
    plt.savefig(outfile, dpi=300)
    plt.close()


def fig_compare_models(outfile: str,
                        models, labels, colors,
                        mse_filepath='valid_avg_mse.parquet'):
    """
    Create a figure comparing average batched MSE among models.
    outfile: file to save as
    model: MAE model (t10, t35, t75)
    labels: labels from pandas frame to plot
    mse_filepath: file with mses
    """
    mse = pd.read_parquet(mse_filepath, engine='pyarrow')
    
    fig, ax = plt.subplots()
    percent = ''
    for l, c in zip(labels, colors):
        x = mse['avg_LL']
        y = mse[l]
        percent = l[-2:]
        plt.scatter(x, y, color=c)

    # plot specifics
    plt.yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
    plt.legend(labels=models, title='Model',
               title_fontsize='small', fontsize='small', fancybox=True, ncol=2)
    plt.title('{}% Masking Comparison'.format(percent))
    plt.xlabel("Average LL Per Batch")
    plt.ylabel("log$_{10}$ MSE")
    
    plt.savefig(outfile, dpi=300)
    plt.close()
    return 0
"""
# This is here for me 
# plot t75
labels = ['avg_mse_t75_p10', 'avg_mse_t75_p20', 'avg_mse_t75_p30', 'avg_mse_t75_p40', 'avg_mse_t75_p50']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
fig_batch_mse("help75.png", "t75", labels, colors)

# plot t10
labels = ['avg_mse_t10_p10', 'avg_mse_t10_p20', 'avg_mse_t10_p30', 'avg_mse_t10_p40', 'avg_mse_t10_p50']
fig_batch_mse("help10.png", "t10", labels, colors)

# plot t35
labels = ['avg_mse_t35_p10', 'avg_mse_t35_p20', 'avg_mse_t35_p30', 'avg_mse_t35_p40', 'avg_mse_t35_p50']
fig_batch_mse("help35.png", "t35", labels, colors)


# plot p10
labels = ['avg_mse_t10_p10', 'avg_mse_t35_p10', 'avg_mse_t75_p10']
colors = ['tab:blue', 'tab:green',  'tab:orange']
models = ['t10', 't35', 't75']
fig_compare_models("compare_p10.png", models, labels, colors)

# plot p20
labels = ['avg_mse_t10_p20', 'avg_mse_t35_p20', 'avg_mse_t75_p20']
fig_compare_models("compare_p20.png", models, labels, colors)

# plot p30
labels = ['avg_mse_t10_p30', 'avg_mse_t35_p30', 'avg_mse_t75_p30']
fig_compare_models("compare_p30.png", models, labels, colors)

#plot p40
labels = ['avg_mse_t10_p40', 'avg_mse_t35_p40', 'avg_mse_t75_p40']
fig_compare_models("compare_p40.png", models, labels, colors)

#plot p50
labels = ['avg_mse_t10_p50', 'avg_mse_t35_p50', 'avg_mse_t75_p50']
fig_compare_models("compare_p50.png", models, labels, colors)

"""
