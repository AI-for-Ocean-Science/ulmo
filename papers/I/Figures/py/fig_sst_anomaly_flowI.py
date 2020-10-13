""" Figures for the first SST OOD paper"""
import os, sys
import numpy as np
import glob


import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

mpl.rcParams['font.family'] = 'stixgeneral'

import healpy as hp
import h5py

import pandas
import seaborn as sns

from ulmo.analysis import cc as ulmo_cc
from ulmo import plotting
from ulmo.utils import image_utils
from ulmo.utils import models as model_utils
from ulmo import defs

from IPython import embed


# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import results

extract_path = defs.extract_path
model_path = defs.model_path
eval_path = defs.eval_path

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


def fig_in_painting(outfile, iexpmle=4, vmnx=(8, 24)):
    """

    Parameters
    ----------
    outfile
    iexpmle
    vmnx

    Returns
    -------

    """
    # Grab it
    field, mask = image_utils.grab_img(iexpmle, 'Extracted')
    masked_field = field.copy()
    masked_field[mask == 1] = -np.nan


    # Plot
    fig = plt.figure(figsize=(10, 4))
    pal, cm = plotting.load_palette()
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # Before in-painting
    ax1 = plt.subplot(gs[0])
    sns.heatmap(masked_field, ax=ax1, xticklabels=[], yticklabels=[], cmap=cm,
                vmin=vmnx[0], vmax=vmnx[1])

    ax2 = plt.subplot(gs[1])
    sns.heatmap(field, ax=ax2, xticklabels=[], yticklabels=[], cmap=cm,
                vmin=vmnx[0], vmax=vmnx[1])

    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_evals_spatial(pproc, outfile, nside=64):
    """

    Parameters
    ----------
    pproc
    outfile
    nside

    Returns
    -------

    """
    # Load
    evals_tbl = results.load_log_prob(pproc)

    # Healpix me
    hp_events = evals_to_healpix(evals_tbl, nside, log=True)

    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    # Median dSST
    hp.mollview(hp_events, min=0, #max=2,
                cmap='Blues',
                flip='geo', title='', unit=r'$\log_{10} \, N_{\rm evals}$',
                rot=(0., 180., 180.))

    # Layout and save
    #plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))



def fig_auto_encode(outfile, iexpmle=4, vmnx=(-5, 5)):

    # Grab it
    field, mask = image_utils.grab_img(iexpmle, 'PreProc')
    fields = np.reshape(field, (1,1,64,64))

    # Load up the model
    pae = model_utils.load('standard')
    # Reconstruct
    recons = pae.reconstruct(fields)

    # Plot
    fig = plt.figure(figsize=(10, 4))
    pal, cm = plotting.load_palette()
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # Original
    ax1 = plt.subplot(gs[0])
    sns.heatmap(field[0,...], ax=ax1, xticklabels=[], yticklabels=[], cmap=cm,
                vmin=vmnx[0], vmax=vmnx[1])

    # Reconstructed
    ax2 = plt.subplot(gs[1])
    sns.heatmap(recons[0,0,...], ax=ax2, xticklabels=[], yticklabels=[], cmap=cm,
                vmin=vmnx[0], vmax=vmnx[1])

    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_LL_SSTa(outfile):

    evals_tbl = results.load_log_prob('std')
    logL = evals_tbl.log_likelihood.values


    # Plot
    fig = plt.figure(figsize=(10, 4))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Original
    ax = plt.subplot(gs[0])

    low_logL = np.quantile(logL, 0.05)
    high_logL = np.quantile(logL, 0.95)
    sns.distplot(logL)
    plt.axvline(low_logL, linestyle='--', c='r')
    plt.axvline(high_logL, linestyle='--', c='r')
    plt.xlabel('Log Likelihood')
    plt.ylabel('Probability Density')

    # Inset for lowest LL
    cut_LL = -1500.
    lowLL = logL < cut_LL
    axins = ax.inset_axes([0.1, 0.4, 0.47, 0.47])
    #axins.scatter(evals_tbl.date.values[lowLL], evals_tbl.log_likelihood.values[lowLL])
    #bins = np.arange(-6000., -1000., 250)
    #out_hist, out_bins = np.histogram(logL[lowLL], bins=bins)
    #embed(header='316 of figs')
    #axins.hist(logL[lowLL], color='k')
    axins.scatter(evals_tbl.log_likelihood.values[lowLL], evals_tbl.date.values[lowLL],
                  s=0.1)
    axins.set_xlim(-5500., cut_LL)
    plt.gcf().autofmt_xdate()


    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))



def evals_to_healpix(eval_tbl, nside, log=False):
    """
    Generate a healpix map of where the input
    MHW Systems are located on the globe

    Parameters
    ----------
    mhw_sys : pandas.DataFrame
    nside : int

    Returns
    -------
    healpix_array : hp.ma

    """
    # Grab lats, lons
    lats = eval_tbl.latitude.values
    lons = eval_tbl.longitude.values

    # Healpix coords
    theta = (90 - lats) * np.pi / 180.
    phi = lons * np.pi / 180.
    idx_all = hp.pixelfunc.ang2pix(nside, theta, phi)

    # Count events
    npix_hp = hp.nside2npix(nside)
    all_events = np.ma.masked_array(np.zeros(npix_hp, dtype='int'))
    for idx in idx_all:
        all_events[idx] += 1

    zero = all_events == 0
    if log:
        all_events[~zero] = np.log10(all_events[~zero])

    # Mask
    hpma = hp.ma(all_events.astype(float))
    hpma.mask = zero

    # Return
    return hpma

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

    # Spatial of all evaluations
    if flg_fig & (2 ** 3):
        for outfile in ['fig_std_evals_spatial.png']:
            fig_evals_spatial('std', outfile)

    # In-painting
    if flg_fig & (2 ** 4):
        for outfile in ['fig_in_painting.png']:
            fig_in_painting(outfile)

    # Auto-encode
    if flg_fig & (2 ** 5):
        for outfile in ['fig_auto_encode.png']:
            fig_auto_encode(outfile)

    # LL for SSTa
    if flg_fig & (2 ** 6):
        for outfile in ['fig_LL_SSTa.png']:
            fig_LL_SSTa(outfile)

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # Month histogram
        #flg_fig += 2 ** 1  # <T> histogram
        #flg_fig += 2 ** 2  # CC
        #flg_fig += 2 ** 3  # All Evals spatial
        #flg_fig += 2 ** 4  # In-painting
        #flg_fig += 2 ** 5  # Auto-encode
        flg_fig += 2 ** 6  # LL SSTa
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
