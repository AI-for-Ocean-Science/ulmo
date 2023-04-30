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
from ulmo import io as ulmo_io

from IPython import embed

'''
sst_path = os.getenv('OS_SST')
ogcm_path = os.getenv('OS_OGCM')
enki_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki')
'''

def fig_batch_rmse(model, rmse_filepath='valid_avg_rms.csv'):
    """
    Creates a figure of average RMSE by LL batches.
    model: MAE model (10, 35, 50, 75)
    rmse_filepath: file with rmse's
    """
    # load rmse
    rmse = pd.read_csv(rmse_filepath)
    #mse = pd.read_parquet(mse_filepath, engine='pyarrow')
    
    # setup
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    fig, ax = plt.subplots()
    
    # Plot
    masks = [10, 20, 30, 40, 50]
    plt_labels = []
    for p, c in zip(masks, colors):
        x = rmse['median_LL']
        y = rmse['rms_t{t}_p{p}'.format(t=model, p=p)]
        plt_labels.append('rms_t{t}_p{p}%'.format(t=model, p=p))
        plt.scatter(x, y, color=c)

    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
    plt.legend(labels=plt_labels, title='Masking Ratio',
               title_fontsize='small', fontsize='small', fancybox=True)
    plt.title('{} Model: Average RMSE Based on LL'.format(model))
    plt.xlabel("Median LL Per Batch")
    plt.ylabel("RMSE")

    # save
    outfile = 'rmse_t{}.png'.format(model)
    plt.savefig(outfile, dpi=300)
    plt.close()

    
#### Old legacy version ###
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


def fig_viirs_rms(outfile: str, t:int=10, p:int=10,
                  mse_filepath='valid_avg_mse.parquet'):
    """
    Create a figure comparing average batched RMS between VIIRS and LLC
    outfile: file to save as
    model: MAE model (t10, t35, t75)
    labels: labels from pandas frame to plot
    mse_filepath: file with mses
    """
    # Load tables
    viirs_file = os.path.join(sst_path, 'VIIRS', 'Tables', 
                              'VIIRS_all_100clear_std.parquet')
    viirs = ulmo_io.load_main_table(viirs_file)

    llc_file = os.path.join(enki_path, 'Tables', 
                              'MAE_LLC_valid_nonoise.parquet')
    llc = ulmo_io.load_main_table(llc_file)

    # Batch me
    percentiles = np.arange(0, 100, 10) + 10
    viirs_per = np.percentile(viirs['LL'], percentiles)

    avg_LL, viirs_rmse, llc_rmse = [], [], []
    # Evaluate
    for ss, LL_per in enumerate(viirs_per):
        if ss == 0:
            LL_min = -1e10
        else:
            LL_min = viirs_per[ss-1]

        # LL
        vidx = (viirs['LL'] <= LL_per) & (viirs['LL'] > LL_min)
        avg_LL.append(np.nanmean(viirs['LL'][vidx]))

        # VIIRS
        viirs_rmse.append(np.nanmedian(viirs[f'RMS_t{t}_p{p}'][vidx]))

        # LLC
        lidx = (llc['LL'] <= LL_per) & (llc['LL'] > LL_min)
        llc_rmse.append(np.nanmedian(llc[f'RMS_t{t}_p{p}'][lidx]))

    # Plot
    ax = plt.gca()

    ax.scatter(avg_LL, viirs_rmse, color='blue', label='VIIRS')
    ax.scatter(avg_LL, llc_rmse, color='red', label='LLC')

    # plot specifics
    #plt.yscale('log')
    #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
    plt.legend(title='Dataset',
               title_fontsize='small', fontsize='small', fancybox=True, ncol=2)
    plt.title(f'VIIRS (and LLC): t={t}, p={p}')
    plt.xlabel("Average LL Per Batch")
    plt.ylabel("RMS")
    
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f'Wrote: {outfile}')
    return

#### ########################## #########################
fig_batch_rmse(75)

"""
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # Explore the bias
    if flg_fig & (2 ** 0):
        fig_viirs_rms('fig_viirs_llc_rms.png')

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        flg_fig += 2 ** 0  # VIIRS vs. LLC MSE
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
"""