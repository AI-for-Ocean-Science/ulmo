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

from IPython import embed

eval_path = os.path.join(os.getenv("SST_OOD"), 'Evaluations')
extract_path = os.path.join(os.getenv("SST_OOD"), 'Extractions')


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

    # Find a good example
    prob_file = os.path.join(eval_path,
                             'MODIS_R2019_2010_95clear_128x128_preproc_std_log_probs.csv')
    print("Grabbing an example")
    df = pandas.read_csv(prob_file)
    cloudy = df.clear_fraction > 0.045
    df = df[cloudy]
    i_LL = np.argsort(df.log_likelihood.values)

    # One, psuedo-random
    example = df.iloc[i_LL[iexpmle]]


    print("Extracting")
    # Grab out of Extraction file
    extract_file = os.path.join(extract_path,
                             'MODIS_R2019_2010_95clear_128x128_inpaintT.h5')
    f = h5py.File(extract_file, mode='r')
    key = 'metadata'
    meta = f[key]
    df_ex = pandas.DataFrame(meta[:].astype(np.unicode_), columns=meta.attrs['columns'])

    imt = (df_ex.filename.values == example.filename) & (
            df_ex.row.values.astype(int) == example.row) & (
            df_ex.column.values.astype(int) == example.column)
    assert np.sum(imt) == 1
    index = df_ex.iloc[imt].index[0]

    # Grab image + mask
    field = f['fields'][index]
    mask = f['masks'][index]

    masked_field = field.copy()
    masked_field[mask == 1] = np.nan

    f.close()


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

    # Load up the tables
    table_files = glob.glob(os.path.join(eval_path, 'R2010_on*{}_log_prob.csv'.format(pproc)))

    # Cut down?
    #table_files = table_files[0:2]

    evals_tbl = pandas.DataFrame()
    for table_file in table_files:
        print("Loading: {}".format(table_file))
        df = pandas.read_csv(table_file)
        evals_tbl = pandas.concat([evals_tbl, df])

    print('NEED TO ADD IN 2010!!!')

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

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # Month histogram
        #flg_fig += 2 ** 1  # <T> histogram
        #flg_fig += 2 ** 2  # CC
        #flg_fig += 2 ** 3  # All Evals spatial
        flg_fig += 2 ** 4  # In-painting
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
