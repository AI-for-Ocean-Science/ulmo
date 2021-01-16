""" Figures for the first SST OOD paper"""
import os, sys
import numpy as np
import glob


import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs

mpl.rcParams['font.family'] = 'stixgeneral'

import healpy as hp
import h5py

import pandas
import seaborn as sns

from ulmo.analysis import cc as ulmo_cc
from ulmo import plotting
from ulmo.utils import image_utils
from ulmo.utils import models as model_utils
from ulmo.utils import utils as utils
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
    """
    CC fraction
    """

    # Build by ulmo/analysis/cc.py
    tst_file = os.path.join(os.getenv('SST_OOD'), 'Analysis', 'cc_2010.h5')

    # Load data
    f = h5py.File(tst_file, mode='r')
    fracCC = f['fracCC'][:]

    # Average
    mean_fCC = np.mean(fracCC, axis=0)
    #embed(header='136 of figs')

    # Differential
    diff_CC = mean_fCC - np.roll(mean_fCC, -1)
    diff_CC[-1] = mean_fCC[-1]
    yzero = np.zeros_like(diff_CC)

    # Figure time
    fig = plt.figure(figsize=(7, 5))
    plt.clf()
    ax = plt.gca()

    # Plot
    p1 = ax.plot(1-ulmo_cc.CC_values, diff_CC, 'o', color='b', label='Fraction')
    #p1 = ax.fill_between(np.array(1-ulmo_cc.CC_values), yzero, diff_CC,
    #                     step='mid',
    #                     alpha=0.5,
    #                     color='blue',
    #                     label='Differential')

    # Labels
    ax.set_ylabel(r'Fraction of Total Images')
    ax.set_xlabel(r'Clear Fraction (CF=1-CC)')
    ax.set_ylim(0., 0.05)
    #ax.set_ylim(0., 1.0)

    # Font size
    fsz = 15.
    set_fontsize(ax, fsz)

    '''
    # Cumulative
    axC = ax.twinx()
    axC.set_ylim(0., 1.)

    p2 = axC.plot(1-ulmo_cc.CC_values[1:], mean_fCC[1:], color='k', label='Cumulative')
    axC.set_ylabel(r'Cumulative Distribution')

    set_fontsize(axC, fsz)

    #ax.set_yscale('log')
    #ax.minorticks_on()

    #plts = p1 + p2
    plts = p2
    labs = [p.get_label() for p in plts]

    legend = plt.legend(plts, labs, loc='upper right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='large', numpoints=1)
    '''

    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def img_exmple(iexmple=4, cherry=False):
    prob_file = os.path.join(eval_path,
                             'MODIS_R2019_2010_95clear_128x128_preproc_std_log_probs.csv')
    table_files = [prob_file]
    # Find a good example
    print("Grabbing an example")
    df = results.load_log_prob('std', table_files=table_files)
    if cherry:
        bools = np.all([df.filename.values == 'AQUA_MODIS.20100619T062008.L2.SST.nc',
                     df.row.values == 253, df.column.values == 924], axis=0)
        icherry = np.where(bools)[0][0]
        # Replace
        example = df.iloc[icherry]
    else:
        cloudy = df.clear_fraction > 0.045
        df = df[cloudy]
        i_LL = np.argsort(df.log_likelihood.values)

        # One, psuedo-random
        example = df.iloc[i_LL[iexmple]]
    return example


def fig_in_painting(outfile, iexmple=4, vmnx=(8, 24)):
    """

    Parameters
    ----------
    outfile
    iexpmle
    vmnx

    Returns
    -------

    """
    example = img_exmple(iexmple=iexmple)

    # Grab it
    field, mask = image_utils.grab_img(example, 'Extracted', ptype='std')
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


def fig_spatial_all(pproc, outfile, nside=64):
    """
    Spatial distribution of the evaluations

    Parameters
    ----------
    pproc
    outfile
    nside

    Returns
    -------

    """
    # Load
    evals_tbl = results.load_log_prob(pproc, feather=True)

    lbl = 'evals'
    use_log = True
    use_mask = True

    # Healpix me
    hp_events, hp_lons, hp_lats = image_utils.evals_to_healpix(
        evals_tbl, nside, log=use_log, mask=use_mask)

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    
    hp.mollview(hp_events, min=0, max=4.,
                hold=True,
                cmap='Blues',
                flip='geo', title='', unit=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl)+'}$',
                rot=(0., 180., 180.))
    #plt.gca().coastlines()

    # Layout and save
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))



def fig_spatial_outliers(pproc, outfile, nside=64):
    """
    Spatial distribution of the evaluations

    Parameters
    ----------
    pproc
    outfile
    nside

    Returns
    -------

    """
    # Load
    evals_tbl = results.load_log_prob(pproc, feather=True)

    cohort = 'outliers'
    point1 = int(0.001 * len(evals_tbl))
    isortLL = np.argsort(evals_tbl.log_likelihood)
    evals_tbl = evals_tbl.iloc[isortLL[0:point1]]
    lbl = 'outliers'
    use_mask = True
    use_log = True

    # Healpix me
    hp_events, hp_lons, hp_lats = image_utils.evals_to_healpix(
        evals_tbl, nside, log=use_log, mask=use_mask)

    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    if cohort == 'all':
        cm = plt.get_cmap('Blues')
        img = ax.tricontourf(hp_lons, hp_lats, hp_events, transform=tformM,
                         levels=20, cmap=cm)#, zorder=10)
    else:
        cm = plt.get_cmap('Reds')
        # Cut
        good = np.invert(hp_events.mask)
        img = plt.scatter(x=hp_lons[good],
            y=hp_lats[good],
            c=hp_events[good],
            cmap=cm,
            s=1,
            transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    clbl=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl)+'}$'
    cb.set_label(clbl, fontsize=20.)
    cb.ax.tick_params(labelsize=17)

    # Coast lines
    if cohort == 'outliers':
        ax.coastlines(zorder=10)
        ax.set_global()

    # Layout and save
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_inlier_vs_outlier(outfile='fig_inlier_vs_outlier.png'):
    """
    Spatial distribution of the evaluations

    Parameters
    ----------
    outfile

    """
    tformP = ccrs.PlateCarree()

    # Load
    evals_tbl = results.load_log_prob('std', feather=True)

    # Add in DT
    if 'DT' not in evals_tbl.keys():
        evals_tbl['DT'] = evals_tbl.T90 - evals_tbl.T10

    # Cut on DT
    cut2 = np.abs(evals_tbl.DT.values-2.) < 0.05
    cut_evals = evals_tbl[cut2].copy()
    lowLL = np.percentile(cut_evals.log_likelihood, 10.)
    hiLL = np.percentile(cut_evals.log_likelihood, 90.)

    low = cut_evals.log_likelihood < lowLL
    high = cut_evals.log_likelihood > hiLL

    fig = plt.figure()#figsize=(14, 8))
    plt.clf()
    ax = plt.axes(projection=tformP)

    # Low
    lw = 0.5
    psize = 5.
    img = plt.scatter(
        x=cut_evals.longitude[low],
        y=cut_evals.latitude[low],
        edgecolors='red',
        facecolors='none',
        s=psize,
        lw=lw,
        transform=tformP)

    # High
    img = plt.scatter(
        x=cut_evals.longitude[high],
        y=cut_evals.latitude[high],
        edgecolors='b',
        facecolors='none',
        s=psize,
        lw=lw,
        transform=tformP)

    # Coast lines
    ax.coastlines(zorder=10)
    ax.set_global()

    # Layout and save
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def tst():
    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.crs as ccrs

    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    ax = plt.axes(projection=ccrs.Mollweide())

    cm = plt.get_cmap('Greens')

    hp_lons = np.random.random(100) * 360 - 180
    hp_lats = np.random.random(100) * 90 - 45
    hp_events = np.random.random(100)

    # Cut down
    img = ax.tricontourf(hp_lons, hp_lats, hp_events, transform=ccrs.PlateCarree(),
                         levels=20, cmap=cm, zorder=10)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    clbl = r'$\log_{10} \, N$'
    cb.set_label(clbl, fontsize=20.)

    ax.coastlines(zorder=10)
    ax.set_global()

    plt.show()


def fig_auto_encode(outfile, iexmple=4, vmnx=(-5, 5)):
    """
    Reconstruction image

    Parameters
    ----------
    outfile
    iexmple
    vmnx

    Returns
    -------

    """
    all_evals_tbl = results.load_log_prob('std', feather=True)
    cherry = np.all([all_evals_tbl.filename.values == 'AQUA_MODIS.20100619T062008.L2.SST.nc',
                     all_evals_tbl.row.values == 253, all_evals_tbl.column.values == 924], axis=0)
    icherry = np.where(cherry)[0][0]
    # Replace
    example = all_evals_tbl.iloc[icherry]

    # Grab it
    field, mask = image_utils.grab_img(example, 'PreProc', ptype='std')
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
    """
    LL distribution

    Parameters
    ----------
    outfile

    Returns
    -------

    """

    evals_tbl = results.load_log_prob('std', feather=True)
    logL = evals_tbl.log_likelihood.values

    isort = np.argsort(logL)
    LL_a = logL[isort[int(len(logL)*0.001)]]

    print("median logL = {}".format(np.median(logL)))

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
    fsz = 17.
    plt.xlabel('Log Likelihood (LL)', fontsize=fsz)
    plt.ylabel('Probability Density', fontsize=fsz)

    # Inset for lowest LL
    cut_LL = LL_a
    lowLL = logL < cut_LL
    axins = ax.inset_axes([0.1, 0.3, 0.57, 0.57])
    #axins.scatter(evals_tbl.date.values[lowLL], evals_tbl.log_likelihood.values[lowLL])
    #bins = np.arange(-6000., -1000., 250)
    #out_hist, out_bins = np.histogram(logL[lowLL], bins=bins)
    #embed(header='316 of figs')
    #axins.hist(logL[lowLL], color='k')
    axins.scatter(evals_tbl.log_likelihood.values[lowLL], 
        evals_tbl.date.values[lowLL], s=0.1)
    #axins.axvline(LL_a, color='k', ls='--')
    axins.set_xlim(-8000., cut_LL)
    axins.minorticks_on()
    axins.set_title('Outliers (lowest 0.1% in LL)')
    plt.gcf().autofmt_xdate()


    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_gallery(outfile, ptype, flavor='outlier'):

    all_evals_tbl = results.load_log_prob(ptype, feather=True)

    # Grab random outliers
    #years = [2008, 2009, 2011, 2012]
    years = np.arange(2003, 2020, 2)
    dyear = 2
    ngallery = 9

    if flavor == 'outlier':
        # Cut
        top = 1000
        isrt = np.argsort(all_evals_tbl.log_likelihood)
        evals_tbl = all_evals_tbl.iloc[isrt[0:top]]
    elif flavor == 'inlier':
        bottom = 1000
        isrt = np.argsort(all_evals_tbl.log_likelihood)
        evals_tbl = all_evals_tbl.iloc[isrt[-bottom:]]
    else:
        raise IOError("Bad flavor")

    gallery_tbl = results.random_imgs(evals_tbl, years, dyear)

    # Over-ride one?
    if flavor == 'outlier' and ptype == 'std':
        # AQUA_MODIS.20100619T062008.L2.SST.nc	253	924	40.497738	-59.93214	0.049987793	20.64104652	15.69499969	23.97500038	22.65999985	18.38500023	-1234.1112
        cherry = np.all([all_evals_tbl.filename.values == 'AQUA_MODIS.20100619T062008.L2.SST.nc',
                  all_evals_tbl.row.values == 253, all_evals_tbl.column.values == 924], axis=0)
        icherry = np.where(cherry)[0][0]
        # Replace
        gallery_tbl.iloc[3] = all_evals_tbl.iloc[icherry]

    if len(gallery_tbl) < ngallery:
        raise ValueError("Uh oh")

    # Plot
    pal, cm = plotting.load_palette()
    fig = plt.figure(figsize=(10, 8))
    plt.clf()
    gs = gridspec.GridSpec(3,3)

    # Original
    for ss in range(ngallery):
        # Axis
        ax = plt.subplot(gs[ss])

        # Grab image
        example = gallery_tbl.iloc[ss]
        field, mask = image_utils.grab_img(example, 'PreProc', ptype=ptype)

        # Plot
        if ptype == 'loggrad':
            vmin, vmax = -5., 0.
        else:
            vmin, vmax = None, None
        sns.heatmap(field[0], ax=ax, xticklabels=[], yticklabels=[], cmap=cm,
                    vmin=vmin, vmax=vmax)

        # Label
        lsz = 17.
        lclr = 'white'
        ax.text(0.05, 0.90, '{}'.format(example.date.strftime('%Y-%m-%d')),
                transform=ax.transAxes, fontsize=lsz, ha='left', color=lclr)
        ax.text(0.05, 0.80, '{:0.3f},{:0.3f}'.format(example.longitude, example.latitude),
                transform=ax.transAxes, fontsize=lsz, ha='left', color=lclr)

    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_LL_vs_DT(ptype, outfile, evals_tbl=None):

    #sns.set_theme()
    #sns.set_style('whitegrid')
    #sns.set_context('paper')


    # Load
    if evals_tbl is None:
        evals_tbl = results.load_log_prob(ptype, feather=True)

    # Add in DT
    if 'DT' not in evals_tbl.keys():
        evals_tbl['DT'] = evals_tbl.T90 - evals_tbl.T10

    # Stats
    cut2 = np.abs(evals_tbl.DT.values-2.) < 0.05
    print("Min LL: {}".format(np.min(evals_tbl.log_likelihood[cut2])))
    print("Max LL: {}".format(np.max(evals_tbl.log_likelihood[cut2])))
    print("Mean LL: {}".format(np.mean(evals_tbl.log_likelihood[cut2])))
    print("RMS LL: {}".format(np.std(evals_tbl.log_likelihood[cut2])))

    # Bins
    bins_LL = np.linspace(-10000., 1100., 22)
    bins_DT = np.linspace(0., 14, 14)

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Total NSpax
    ax_tot = plt.subplot(gs[0])

    jg = sns.jointplot(data=evals_tbl, x='DT', y='log_likelihood',
        kind='hist', bins=200, marginal_kws=dict(bins=200))

    #jg.ax_marg_x.set_xlim(8, 10.5)
    #jg.ax_marg_y.set_ylim(0.5, 2.0)
    jg.ax_joint.set_xlabel(r'$\Delta T$ (K)')
    jg.ax_joint.set_ylabel(r'LL')
    jg.ax_joint.minorticks_on()

    #jg.ax_joint.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    #jg.ax_joint.xaxis.set_major_locator(plt.MultipleLocator(1.0)

    # 2D hist
    #hist2d(evals_tbl.log_likelihood.values, evals_tbl.DT.values,
    #       bins=[bins_LL, bins_DT], ax=ax_tot, color='b')

    #ax_tot.set_xlabel('LL')
    #ax_tot.set_ylabel(r'$\Delta T$')
    #ax_tot.set_ylim(0.3, 5.0)
    #ax_tot.minorticks_on()

    #legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
    #                    handletextpad=0.3, fontsize=19, numpoints=1)

    #set_fontsize(ax_tot, 19.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_LL_vs_LL(outfile, evals_tbl_std=None, evals_tbl_grad=None):


    # Load
    if evals_tbl_std is None:
        evals_tbl_std = results.load_log_prob('std', feather=True)
    if evals_tbl_grad is None:
        evals_tbl_grad = results.load_log_prob('loggrad', feather=True)

    # Outliers
    point1 = int(0.001 * len(evals_tbl_std))
    isortLL_std = np.argsort(evals_tbl_std.log_likelihood)
    outliers_std = evals_tbl_std.iloc[isortLL_std[0:point1]]

    isortLL_grad = np.argsort(evals_tbl_grad.log_likelihood)
    outliers_grad = evals_tbl_grad.iloc[isortLL_grad[0:point1]]

    # Std to grad
    mtchs = utils.match_ids(outliers_std.UID, evals_tbl_grad.UID, require_in_match=False)
    gd_LL = mtchs >= 0
    LL_grad_std = evals_tbl_grad.log_likelihood.values[mtchs[gd_LL]]
    LL_std = outliers_std.log_likelihood.values[gd_LL]

    mtchs2 = utils.match_ids(outliers_grad.UID, evals_tbl_std.UID, require_in_match=False)
    gd_LL2 = mtchs2 >= 0
    LL_std_grad = evals_tbl_std.log_likelihood.values[mtchs2[gd_LL2]]
    LL_grad = outliers_grad.log_likelihood.values[gd_LL2]

    '''
    # Grab em
    LL_grad = []
    for kk in range(len(outliers_std)):
        iobj = outliers_std.iloc[kk]
        gdate = evals_tbl_grad.date == iobj.date
        grow = evals_tbl_grad.row == iobj.row
        gcol = evals_tbl_grad.column == iobj.column
        idx = np.where(gdate & grow & gcol)[0]
        if len(idx) == 1:
            LL_grad.append(evals_tbl_grad.iloc[idx].log_likelihood.values[0])
        else:
            LL_grad.append(np.nan)
    '''

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(2,1)

    #
    ax_std = plt.subplot(gs[0])
    ax_std.scatter(LL_std, LL_grad_std, s=0.2)
    ax_std.set_xlabel('LL SSTa 0.1% Outliers')
    ax_std.set_ylabel('LL_grad')

    ax_log = plt.subplot(gs[1])
    ax_log.scatter(LL_grad, LL_std_grad, s=0.2)
    ax_log.set_xlabel(r'LL $\nabla$SST 0.1% Outliers')
    ax_log.set_ylabel('LL_std')

    set_fontsize(ax_std, 19.)
    set_fontsize(ax_log, 19.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))



def fig_year_month(outfile, ptype, evals_tbl=None, frac=False,
                   all=False):
    """
    Time evolution in outliers

    Parameters
    ----------
    outfile
    ptype
    evals_tbl

    Returns
    -------

    """

    # Load
    if evals_tbl is None:
        evals_tbl = results.load_log_prob(ptype, feather=True)
        print("Loaded..")

    # Outliers
    point1 = int(0.001 * len(evals_tbl))
    isortLL = np.argsort(evals_tbl.log_likelihood)
    outliers = evals_tbl.iloc[isortLL[0:point1]]

    # All
    if all or frac:
        all_years = [item.year for item in evals_tbl.date]
        all_months = [item.month for item in evals_tbl.date]

    # Parse
    years = [item.year for item in outliers.date]
    months = [item.month for item in outliers.date]

    # Histogram
    bins_year = np.arange(2002.5, 2020.5)
    bins_month = np.arange(0.5, 13.5)

    counts, xedges, yedges = np.histogram2d(months, years,
                                            bins=(bins_month, bins_year))
    if all or frac:
        all_counts, _, _ = np.histogram2d(all_months, all_years,
                                            bins=(bins_month, bins_year))

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(5,6)

    # Total NSpax
    ax_tot = plt.subplot(gs[1:,1:-1])

    cm = plt.get_cmap('Blues')
    if frac:
        values  = counts.transpose()/all_counts.transpose()
        lbl = 'Fraction'
    elif all:
        cm = plt.get_cmap('Greens')
        norm = np.sum(all_counts) / np.product(all_counts.shape)
        values = all_counts.transpose()/norm
        lbl = 'Fraction (all)'
    else:
        values = counts.transpose()
        lbl = 'Counts'
    mplt = ax_tot.pcolormesh(xedges, yedges, values, cmap=cm)

    # Color bar
    cbaxes = fig.add_axes([0.03, 0.1, 0.05, 0.7])
    cb = plt.colorbar(mplt, cax=cbaxes, aspect=20)
    #cb.set_label(lbl, fontsize=20.)
    cbaxes.yaxis.set_ticks_position('left')
    cbaxes.set_xlabel(lbl, fontsize=15.)

    ax_tot.set_xlabel('Month')
    ax_tot.set_ylabel('Year')

    set_fontsize(ax_tot, 19.)

    # Edges
    fsz = 15.
    months = np.mean(values, axis=0)
    ax_m = plt.subplot(gs[0,1:-1])
    ax_m.step(np.arange(12)+1, months, color='k', where='mid')
    set_fontsize(ax_m, fsz)
    #ax_m.minorticks_on()

    years = np.mean(values, axis=1)
    ax_y = plt.subplot(gs[1:,-1])
    ax_y.invert_xaxis()
    ax_y.step(years, 2003 + np.arange(17), color='k', where='mid')
    ax_y.set_xlim(40,80)
    set_fontsize(ax_y, fsz)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
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

def hist2d(x, y, bins=20, range=None, weights=None, levels=None, smooth=None,
           ax=None, color=None, plot_datapoints=True, plot_density=True,
           plot_contours=True, no_fill_contours=False, fill_contours=False,
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
           **kwargs):
    """
    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x, y : array_like (nsamples,)
       The samples.

    bins : int or list

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes (optional)
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool (optional)
        Draw the individual data points.

    plot_density : bool (optional)
        Draw the density colormap.

    plot_contours : bool (optional)
        Draw the contours.

    no_fill_contours : bool (optional)
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool (optional)
        Fill the contours.

    contour_kwargs : dict (optional)
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict (optional)
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict (optional)
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.
    """
    from matplotlib.colors import LinearSegmentedColormap, colorConverter
    from scipy.ndimage import gaussian_filter

    if ax is None:
        ax = plt.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=range, weights=weights)
    except ValueError:
        embed(header='732 of figs')
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "'range' argument.")

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        print("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ])

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlim(range[0])
    ax.set_ylim(range[1])

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
            fig_spatial_all('std', outfile)
        fig_spatial_outliers('std', 'fig_std_outliers_spatial.png')

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

    # Outlier gallery
    if flg_fig & (2 ** 7):
        # Outlier
        #for ptype, outfile in zip(['std', 'loggrad'], ['fig_gallery_std.png', 'fig_gallery_loggrad.png']):
        for ptype, outfile in zip(['std'], ['fig_gallery_std.png']):
            fig_gallery(outfile, ptype)
        # Inlier
        #for ptype, outfile in zip(['std', 'loggrad'], ['fig_inlier_gallery_std.png', 'fig_inlier_gallery_loggrad.png']):
        #    fig_gallery(outfile, ptype, flavor='inlier')


    # LL vs LL
    if flg_fig & (2 ** 8):
        for outfile in ['fig_LL_vs_LL.png']:
            fig_LL_vs_LL(outfile)

    # Year, Month
    if flg_fig & (2 ** 9):
        # Counts
        #for ptype, outfile in zip(['std', 'loggrad'], ['fig_year_month_std.png', 'fig_year_month_loggrad.png']):
        for ptype, outfile in zip(['std'], ['fig_year_month_std.png']):
                fig_year_month(outfile, ptype)
        # Fractional
        #for ptype, outfile in zip(['std'], ['fig_year_month_std_frac.png']):
        #    fig_year_month(outfile, ptype, frac=True)
        # All
        #for ptype, outfile in zip(['std'], ['fig_year_month_std_all.png']):
        #    fig_year_month(outfile, ptype, all=True)


    # LL vs. DT
    if flg_fig & (2 ** 11):
        #for ptype, outfile in zip(['std', 'loggrad'],
        #                          ['fig_LL_vs_T_std.png',
        #                           'fig_LL_vs_T_loggrad.png']):
        for ptype, outfile in zip(['std'], ['fig_LL_vs_T_std.png']):
            fig_LL_vs_DT(ptype, outfile)

    # Spatial of all evaluations
    if flg_fig & (2 ** 12):
        fig_inlier_vs_outlier()


    # LL vs. DT
    if flg_fig & (2 ** 20):
        tst()

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # Month histogram
        #flg_fig += 2 ** 1  # <T> histogram
        #flg_fig += 2 ** 2  # CC fractions
        #flg_fig += 2 ** 3  # All Evals spatial
        #flg_fig += 2 ** 4  # In-painting
        #flg_fig += 2 ** 5  # Auto-encode
        flg_fig += 2 ** 6  # LL SSTa
        #flg_fig += 2 ** 7  # Gallery
        #flg_fig += 2 ** 8  # LL_SST vs. LL_grad
        #flg_fig += 2 ** 9  # year, month
        #flg_fig += 2 ** 11  # LL vs DT
        #flg_fig += 2 ** 12  # inlier vs outlier for DT = 2
        #flg_fig += 2 ** 20  # tst
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)

