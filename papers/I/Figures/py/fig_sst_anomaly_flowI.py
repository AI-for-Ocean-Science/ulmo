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

    tst_file = os.path.join(os.getenv('SST_OOD'), 'Analysis', 'cc_2010.h5')

    # Load data
    f = h5py.File(tst_file, mode='r')
    fracCC = f['fracCC'][:]

    # Average
    mean_fCC = np.mean(fracCC, axis=0)

    # Differential
    diff_CC = mean_fCC - np.roll(mean_fCC, -1)
    diff_CC[-1] = mean_fCC[-1]
    yzero = np.zeros_like(diff_CC)

    # Figure time
    fig = plt.figure(figsize=(7, 5))
    plt.clf()
    ax = plt.gca()

    # Plot
    #p1 = ax.plot(1-ulmo_cc.CC_values, diff_CC, label='Differential')
    p1 = ax.fill_between(np.array(1-ulmo_cc.CC_values), yzero, diff_CC,
                         step='mid',
                         alpha=0.5,
                         color='blue',
                         label='Differential')

    # Labels
    ax.set_ylabel(r'Differential Fraction')
    ax.set_xlabel(r'Clear Fraction (1-CC)')
    ax.set_ylim(0., 0.04)

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

    #plts = p1 + p2
    plts = p2
    labs = [p.get_label() for p in plts]

    legend = plt.legend(plts, labs, loc='upper left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='large', numpoints=1)

    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def img_exmple(iexmple=4):
    prob_file = os.path.join(eval_path,
                             'MODIS_R2019_2010_95clear_128x128_preproc_std_log_probs.csv')
    table_files = [prob_file]
    # Find a good example
    print("Grabbing an example")
    df = results.load_log_prob('std', table_files=table_files)
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



def fig_auto_encode(outfile, iexmple=4, vmnx=(-5, 5)):

    example = img_exmple(iexmple=iexmple)
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
    axins = ax.inset_axes([0.1, 0.3, 0.57, 0.57])
    #axins.scatter(evals_tbl.date.values[lowLL], evals_tbl.log_likelihood.values[lowLL])
    #bins = np.arange(-6000., -1000., 250)
    #out_hist, out_bins = np.histogram(logL[lowLL], bins=bins)
    #embed(header='316 of figs')
    #axins.hist(logL[lowLL], color='k')
    axins.scatter(evals_tbl.log_likelihood.values[lowLL], evals_tbl.date.values[lowLL],
                  s=0.1)
    axins.set_xlim(-8000., cut_LL)
    plt.gcf().autofmt_xdate()


    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_gallery(outfile, ptype):

    evals_tbl = results.load_log_prob(ptype)

    # Grab random outliers
    #years = [2008, 2009, 2011, 2012]
    years = np.arange(2003, 2020, 2)
    dyear = 2
    ngallery = 9
    gallery_tbl = results.random_imgs(evals_tbl, years, dyear)

    if len(gallery_tbl) < ngallery:
        raise ValueError("Uh oh")

    # Plot
    pal, cm = plotting.load_palette()
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    gs = gridspec.GridSpec(3,3)

    # Original
    for ss in range(ngallery):
        # Axis
        ax = plt.subplot(gs[ss])

        # Grab image
        example = gallery_tbl.iloc[ss]
        field, mask = image_utils.grab_img(example, 'PreProc', ptype='std')

        # Plot
        sns.heatmap(field[0], ax=ax, xticklabels=[], yticklabels=[], cmap=cm)
                #vmin=vmnx[0], vmax=vmnx[1])

        # Label
        lsz = 17.
        lclr = 'k'
        ax.text(0.05, 0.90, '{}'.format(example.date.strftime('%Y-%m-%d')),
                transform=ax.transAxes, fontsize=lsz, ha='left', color=lclr)
        ax.text(0.05, 0.80, '{:0.3f},{:0.3f}'.format(example.longitude, example.latitude),
                transform=ax.transAxes, fontsize=lsz, ha='left', color=lclr)

    # Layout and save
    # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_LL_vs_DT(outfile, evals_tbl=None, ptype='std'):

    # Load
    if evals_tbl is None:
        evals_tbl = results.load_log_prob(ptype)

    # Add in DT
    if 'DT' not in evals_tbl.keys():
        evals_tbl['DT'] = evals_tbl.T90 - evals_tbl.T10

    # Bins
    bins_LL = np.linspace(-10000., 1100., 22)
    bins_DT = np.linspace(0., 14, 14)

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Total NSpax
    ax_tot = plt.subplot(gs[0])

    # 2D hist
    hist2d(evals_tbl.log_likelihood.values, evals_tbl.DT.values,
           bins=[bins_LL, bins_DT], ax=ax_tot, color='b')

    ax_tot.set_xlabel('LL')
    ax_tot.set_ylabel(r'$\Delta T$')
    #ax_tot.set_ylim(0.3, 5.0)
    #ax_tot.minorticks_on()

    #legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
    #                    handletextpad=0.3, fontsize=19, numpoints=1)

    set_fontsize(ax_tot, 19.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
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

    # Outlier gallery
    if flg_fig & (2 ** 7):
        for ptype, outfile in zip(['std', 'loggrad'], ['fig_gallery_std.png', 'fig_gallery_std.png']):
            fig_gallery(outfile, ptype)

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
        #flg_fig += 2 ** 6  # LL SSTa
        flg_fig += 2 ** 7  # Gallery
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
