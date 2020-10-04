""" Figures for the first SST OOD paper"""
import os, sys
import numpy as np

from datetime import date

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

mpl.rcParams['font.family'] = 'stixgeneral'

import healpy as hp
import h5py

import pandas

from ulmo.analysis import cc as ulmo_cc

from IPython import embed


def fig_db_by_month(outfile):

    # Load db
    anom_db = pandas.read_hdf('../Analysis/MODIS_2010_100clear_48x48_log_probs.hdf')

    N10 = int(np.round(0.1*len(anom_db)))
    i10 = np.argsort(anom_db.log_likelihood.values)[0:N10]
    ih10 = np.argsort(anom_db.log_likelihood.values)[-N10:]

    # Months
    months = np.array([idate.month for idate in anom_db.date])

    # Bin em
    ibins = np.arange(14)
    H_all, bins = np.histogram(months, bins=ibins)
    bincentres = [(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)]
    H_10, _ = np.histogram(months[i10], bins=ibins) # Outliers
    H_h10, _ = np.histogram(months[ih10], bins=ibins) # Inliers

    # Figure time
    fig = plt.figure(figsize=(7, 5))
    plt.clf()
    ax = plt.gca()

    for H, clr, cat in zip([H_all, H_10, H_h10], ['k', 'r', 'b'], ['All', 'Lowest 10%', 'Highest 10%']):
        plt.step(bincentres, H, where='mid', color=clr, label='{}'.format(cat))

    # Labels
    ax.set_ylabel(r'$N$')
    ax.set_xlabel('Month')
    #ax.set_yscale('log')
    ax.minorticks_on()

    legend = plt.legend(loc='lower right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='large', numpoints=1)

    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_db_by_meanT(outfile):

    # Load db
    anom_db = pandas.read_hdf('../Analysis/MODIS_2010_100clear_48x48_log_probs.hdf')

    N10 = int(np.round(0.1*len(anom_db)))
    i10 = np.argsort(anom_db.log_likelihood.values)[0:N10]
    ih10 = np.argsort(anom_db.log_likelihood.values)[-N10:]

    # Months
    avgT = anom_db.mean_temperature.values

    # Bin em
    ibins = np.arange(0, 40, 5)
    H_all, bins = np.histogram(avgT, bins=ibins)
    bincentres = [(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)]
    H_10, _ = np.histogram(avgT[i10], bins=ibins) # Outliers
    H_h10, _ = np.histogram(avgT[ih10], bins=ibins) # Inliers

    # Figure time
    fig = plt.figure(figsize=(7, 5))
    plt.clf()
    ax = plt.gca()

    for H, clr, cat in zip([H_all, H_10, H_h10], ['k', 'r', 'b'], ['All', 'Lowest 10%', 'Highest 10%']):
        plt.step(bincentres, H, where='mid', color=clr, label='{}'.format(cat))

    # Labels
    ax.set_ylabel(r'$N$')
    ax.set_xlabel(r'$<T>$ (C)')
    #ax.set_yscale('log')
    ax.minorticks_on()

    legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='large', numpoints=1)

    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_CC(outfile):

    tst_file = os.path.join(os.getenv('SST_OOD'), 'tst_CC.h5')

    # Load data
    f = h5py.File(tst_file, mode='r')
    fracCC = f['fracCC'][:]

    # Average
    mean_fCC = np.mean(fracCC, axis=0)

    # Differential
    diff_CC = mean_fCC - np.roll(mean_fCC, -1)
    diff_CC[-1] = mean_fCC[-1]

    # Figure time
    fig = plt.figure(figsize=(7, 5))
    plt.clf()
    ax = plt.gca()

    # Plot
    p1 = ax.step(1-ulmo_cc.CC_values, diff_CC, label='Differential')

    # Labels
    ax.set_ylabel(r'Differential Fraction')
    ax.set_xlabel(r'Clear Fraction (1-CC)')
    ax.set_ylim(0., 0.05)

    # Cumulative
    axC = ax.twinx()
    axC.set_ylim(0., 1.)

    p2 = axC.plot(1-ulmo_cc.CC_values, mean_fCC, color='k', label='Cumulative')
    axC.set_ylabel(r'Cumulative Fraction')

    # Font sizes
    fsz = 15.
    set_fontsize(ax, fsz)
    set_fontsize(axC, fsz)

    #ax.set_yscale('log')
    #ax.minorticks_on()

    plts = p1 + p2
    labs = [p.get_label() for p in plts]

    legend = plt.legend(plts, labs, loc='upper left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='large', numpoints=1)

    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
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

    # Month histogram
    if flg_fig & (2 ** 0):
        for outfile in ['fig_db_by_month.png', 'fig_db_by_month.pdf']:
            fig_db_by_month(outfile)

    # <T> histogram
    if flg_fig & (2 ** 1):
        for outfile in ['fig_db_by_meanT.png', 'fig_db_by_meanT.pdf']:
            fig_db_by_meanT(outfile)

    # CC figure
    if flg_fig & (2 ** 2):
        for outfile in ['fig_CC.png']: #, 'fig_CC.pdf']:
            fig_CC(outfile)

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # Month histogram
        #flg_fig += 2 ** 1  # <T> histogram
        flg_fig += 2 ** 2  # CC
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
