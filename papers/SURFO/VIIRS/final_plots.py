# Final Figures for SURFO Paper
import os, sys


#imports
import pandas
import healpy as hp
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

mpl.rcParams['font.family'] = 'stixgeneral'

from ulmo.utils import image_utils


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
    isortLL = np.argsort(evals_tbl.LL)
    outliers = evals_tbl.iloc[isortLL[0:point1]]

    # All
    if all or frac:
        all_years = [item.year for item in evals_tbl.datetime]
        all_months = [item.month for item in evals_tbl.datetime]

    # Parse
    years = [item.year for item in outliers.datetime]
    months = [item.month for item in outliers.datetime]

    # Histogram
    bins_year = np.arange(2012.5, 2021.5)
    bins_month = np.arange(0.5, 13.5)

    counts, xedges, yedges = np.histogram2d(months, years,
                                            bins=(bins_month, bins_year))
    if all or frac:
        all_counts, _, _ = np.histogram2d(all_months, all_years,
                                            bins=(bins_month, bins_year))

    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    gs = plt.GridSpec(5,6)

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

    #ax_tot.set_fontsize(ax_tot, 19.)

    # Edges
    fsz = 30.
    months = np.mean(values, axis=0)
    ax_m = plt.subplot(gs[0,1:-1])
    ax_m.step(np.arange(12)+1, months, color='k', where='mid')
    #ax_m.set_fontsize(ax_m, fsz)
    #ax_m.minorticks_on()

    years = np.mean(values, axis=1)
    ax_y = plt.subplot(gs[1:,-1])
    ax_y.invert_xaxis()
    ax_y.step(years, 2013 + np.arange(8), color='k', where='mid')
    #ax_y.set_xlim(40,80)
    #ax_y.set_fontsize(ax_y, fsz)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.1)
    plt.show
    plt.savefig('month_by_year_of_'+ str(outfile), dpi=600)
    #plt.close()
    #print('Wrote {:s}'.format(outfile))
    
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
    print("Min LL: {}".format(np.min(evals_tbl.LL[cut2])))
    print("Max LL: {}".format(np.max(evals_tbl.LL[cut2])))
    print("Mean LL: {}".format(np.mean(evals_tbl.LL[cut2])))
    print("RMS LL: {}".format(np.std(evals_tbl.LL[cut2])))

    # Bins
    bins_LL = np.linspace(-10000., 1100., 22)
    bins_DT = np.linspace(0., 14, 14)

    fig = plt.figure(figsize=(12, 8))
    #plt.clf()
    #gs = plt.GridSpec(1,1)

    # Total NSpax
    #ax_tot = plt.subplot(gs[0])

    jg = sns.jointplot(data=evals_tbl, x='DT', y='LL',
        kind='hist', bins=200, marginal_kws=dict(bins=200))

    #jg.ax_marg_x.set_xlim(8, 10.5)
    #jg.ax_marg_y.set_ylim(0.5, 2.0)
    jg.ax_joint.set_xlabel(r'$\Delta T$ (K)')
    jg.ax_joint.set_ylabel(r'LL')
    xmnx = (0., 14.5)
    jg.ax_joint.set_xlim(xmnx[0], xmnx[1])
    #ymnx = (-11400., 1700)
    #jg.ax_joint.set_ylim(ymnx[0], ymnx[1])
    jg.ax_joint.minorticks_on()

    # Horizontal line
    lowLL_val = np.percentile(evals_tbl.LL, 0.1)
    jg.ax_joint.plot(xmnx, [lowLL_val]*2, '--', color='gray')
    
    '''
    # Vertical lines
    jg.ax_joint.plot([2.]*2, ymnx, '-', color='gray', lw=1)
    jg.ax_joint.plot([2.1]*2, ymnx, '-', color='gray', lw=1)
    '''

    #set_fontsize(jg.ax_joint, 17.)

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
    plt.savefig('dt_vs_LL_of_'+ str(outfile), dpi=1200)
    plt.show()
    #plt.close()
    print('Wrote {:s}'.format(outfile))
    
    
def evals_to_healpix(eval_tbl, nside, mask=True):
    """
    Generate a healpix map of where the input
    MHW Systems are located on the globe
    Parameters
    ----------
    mhw_sys : pandas.DataFrame
    nside : int  # nside is a number that sets the resolution of map
    mask : bool, optional
    Returns
    -------
    healpix_array : hp.ma (number of cutouts)
    lats : np.ndarray
    lons : np.ndarray
    healpix_array : hp.ma (average LL)
    """
    # Grab lats, lons
    lats = eval_tbl.lat.values
    lons = eval_tbl.lon.values

    # Grab LL values
    vals = eval_tbl.LL.values

    # Healpix coords
    theta = (90 - lats) * np.pi / 180. 
    phi = lons * np.pi / 180.
    idx_all = hp.pixelfunc.ang2pix(nside, theta, phi) 

    # Count events
    npix_hp = hp.nside2npix(nside)
    all_events = np.ma.masked_array(np.zeros(npix_hp, dtype='int')) 
    all_values = np.ma.masked_array(np.zeros(npix_hp, dtype='int')) 

    for i, idx in enumerate(idx_all):
        all_events[idx] += 1
        all_values[idx] += vals[i] 

    zero = all_events == 0 
    
    float_events = all_events.astype(float)
    float_values = all_values.astype(float)
    float_values[~zero] = all_values[~zero]/all_events[~zero]


    # Mask
    hpma = hp.ma(float_events)
    hpma1 = hp.ma(float_values)
    if mask:  # if you want to mask float_events
        hpma.mask = zero # current mask set to zero array, where Trues (no events) are masked
        hpma1.mask = zero 

    # Angles
    hp_lons, hp_lats = hp.pixelfunc.pix2ang(nside, np.arange(npix_hp), lonlat=True)

    # Return
    return hpma, hp_lons, hp_lats, hpma1

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
    
def show_spatial_diff(sub_tbl1:pandas.DataFrame, sub_tbl2:pandas.DataFrame,
                 nside=64, use_log=True, 
                 use_mask=True,
                 lbl1=None, lbl2=None, lbl3=None, figsize=(24,16), 
                 color1='Blues', color2='Oranges', color3='Greys', show=True):
    """Generate a global map of the location of the input
    cutouts
    Args:
        main_tbl (pandas.DataFrame): table of cutouts
        nside (int, optional): [description]. Defaults to 64.
        use_log (bool, optional): [description]. Defaults to True.
        use_mask (bool, optional): [description]. Defaults to True.
        lbl ([type], optional): [description]. Defaults to None.
        figsize (tuple, optional): [description]. Defaults to (12,8).
        color (str, optional): [description]. Defaults to 'Reds'.
        show (bool, optional): If True, show on the screen.  Defaults to True
    Returns:
        matplotlib.Axis: axis holding the plot
    """
    # Healpix me
    hp_events1, hp_lons1, hp_lats1 = image_utils.evals_to_healpix(
        sub_tbl1, nside, log=use_log, mask=use_mask)
    
    hp_events2, hp_lons2, hp_lats2 = image_utils.evals_to_healpix(
        sub_tbl2, nside, log=use_log, mask=use_mask)
    
    # Figure
    
    fig = plt.figure(figsize=figsize)
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm1 = plt.get_cmap(color1)
    cm2 = plt.get_cmap(color2)
    cm3 = plt.get_cmap(color3)
    
    # Cut
    good1 = np.invert(hp_events1.mask)
    img1 = plt.scatter(x=hp_lons1[good1],
        y=hp_lats1[good1],
        c=hp_events1[good1], 
        cmap=cm1,
        s=1,
        transform=tformP)
    
    good2 = np.invert(hp_events2.mask)
    img2 = plt.scatter(x=hp_lons2[good2],
        y=hp_lats2[good2],
        c=hp_events2[good2], 
        cmap=cm2,
        s=1,
        transform=tformP)
    
    both = np.invert((hp_events2.mask==False) & (hp_events1.mask ==False))
    img3 = plt.scatter(x=hp_lons2[both],
        y=hp_lats2[both],
        c=hp_events2[both],
        cmap=cm3,
        s=1,
        transform=tformP) 
    #c=hp_events2[both]
    

    # Colorbar
    cb1 = plt.colorbar(img1, orientation='vertical', location = 'left', shrink = 0.4,pad=0.02)
    cb2 = plt.colorbar(img2, orientation='vertical', location = 'right',shrink = 0.4,pad=0.02)
    cb3 = plt.colorbar(img3, orientation='horizontal', location = 'bottom',shrink = 0.5,pad=0.02)
    
    if lbl1 is not None:
        clbl1=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl1)+'}$'
        cb1.set_label(clbl1, fontsize=20.)
    cb1.ax.tick_params(labelsize=17)
    
    if lbl2 is not None:
        clbl2=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl2)+'}$'
        cb2.set_label(clbl2, fontsize=20.)
    cb2.ax.tick_params(labelsize=17)
    
    if lbl3 is not None:
        clbl3=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl3)+'}$'
        cb3.set_label(clbl3, fontsize=20.)
    cb3.ax.tick_params(labelsize=17)
    

    # Coast lines
    ax.coastlines(zorder=10)
    ax.set_global()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
        color='black', alpha=0.5, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right=False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black'}# 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black'}# 'weight': 'bold'}


    # Layout and save
    plt.show()
        

    return ax


def show_spatial_LL_diff(main_tbl:pandas.DataFrame,
                 nside=64, use_log=True, 
                 use_mask=True,
                 lbl1='Lower10', lbl2='Higer10', lbl3='Both', figsize=(24,16), 
                 color1='Blues', color2='Oranges', color3='Greys', show=True):
    """Generate a global map of the location of the input
    cutouts
    Args:
        sub_tbl1: table of cutouts
        sub_tbl2: table of cutouts to compare to tbl1
        nside (int, optional): [description]. Defaults to 64.
        use_log (bool, optional): [description]. Defaults to True.
        use_mask (bool, optional): [description]. Defaults to True.
        lbl ([type], optional): [description]. Defaults to None.
        figsize (tuple, optional): [description]. Defaults to (12,8).
        color (str, optional): [description]. Defaults to 'Oranges'.
        show (bool, optional): If True, show on the screen.  Defaults to True
    Returns:
        matplotlib.Axis: axis holding the plot
    """
    N = int(np.round(0.1*len(main_tbl)))
    sub_tbl1 = main_tbl[:N]
    sub_tbl2 = main_tbl[-N:]
    
    # Healpix me
    hp_events1, hp_lons1, hp_lats1 = image_utils.evals_to_healpix(
        sub_tbl1, nside, log=use_log, mask=use_mask)
    
    hp_events2, hp_lons2, hp_lats2 = image_utils.evals_to_healpix(
        sub_tbl2, nside, log=use_log, mask=use_mask)
    
    # Figure
    
    fig = plt.figure(figsize=figsize)
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm1 = plt.get_cmap(color1)
    cm2 = plt.get_cmap(color2)
    cm3 = plt.get_cmap(color3)
    
    # Cut
    good1 = np.invert(hp_events1.mask)
    img1 = plt.scatter(x=hp_lons1[good1],
        y=hp_lats1[good1],
        c=hp_events1[good1], 
        cmap=cm1,
        s=1,
        transform=tformP)
    
    good2 = np.invert(hp_events2.mask)
    img2 = plt.scatter(x=hp_lons2[good2],
        y=hp_lats2[good2],
        c=hp_events2[good2], 
        cmap=cm2,
        s=1,
        transform=tformP)
    
    both = np.invert((hp_events2.mask==False) & (hp_events1.mask ==False))
    img3 = plt.scatter(x=hp_lons2[both],
        y=hp_lats2[both],
        c=hp_events2[both],
        cmap=cm3,
        s=1,
        transform=tformP) 
    

    # Colorbar
    cb1 = plt.colorbar(img1, orientation='vertical', location = 'left',shrink=0.45,pad=0.03)
    cb2 = plt.colorbar(img2, orientation='vertical', location = 'right',shrink=0.45,pad=0.03)
    cb3 = plt.colorbar(img3, orientation='horizontal', location = 'bottom',pad=0.01)
    
    if lbl1 is not None:
        clbl1=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl1)+'}$'
        cb1.set_label(clbl1, fontsize=20.)
    cb1.ax.tick_params(labelsize=17)
    
    if lbl2 is not None:
        clbl2=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl2)+'}$'
        cb2.set_label(clbl2, fontsize=20.)
    cb2.ax.tick_params(labelsize=17)
    
    if lbl3 is not None:
        clbl3=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl3)+'}$'
        cb3.set_label(clbl3, fontsize=20.)
    cb3.ax.tick_params(labelsize=17)
    

    # Coast lines
    ax.coastlines(zorder=10)
    ax.set_global()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
        color='black', alpha=0.5, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right=False
    gl.xlines = True
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black'}# 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black'}# 'weight': 'bold'}


    # Layout and save
    plt.show
    #plt.savefig('month_by_year_of_'+ str(outfile), dpi=600)

    return ax



def show_spatial(main_tbl:pandas.DataFrame, 
                 nside=64, use_log=True, 
                 use_mask=True, tricontour=False,
                 lbl=None, figsize=(12,8), 
                 color='Reds', show=True):
    """Generate a global map of the location of the input
    cutouts

    Args:
        main_tbl (pandas.DataFrame): table of cutouts
        nside (int, optional): [description]. Defaults to 64.
        use_log (bool, optional): [description]. Defaults to True.
        use_mask (bool, optional): [description]. Defaults to True.
        tricontour (bool, optional): [description]. Defaults to False.
        lbl ([type], optional): [description]. Defaults to None.
        figsize (tuple, optional): [description]. Defaults to (12,8).
        color (str, optional): [description]. Defaults to 'Reds'.
        show (bool, optional): If True, show on the screen.  Defaults to True

    Returns:
        matplotlib.Axis: axis holding the plot
    """
    # Healpix me
    hp_events, hp_lons, hp_lats = image_utils.evals_to_healpix(
        main_tbl, nside, log=use_log, mask=use_mask)
    
    # Figure
    
    fig = plt.figure(figsize=figsize)
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    if tricontour:
        cm = plt.get_cmap(color)
        img = ax.tricontourf(hp_lons, hp_lats, hp_events, transform=tformM,
                         levels=20, cmap=cm)#, zorder=10)
    else:
        cm = plt.get_cmap(color)
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
    if lbl is not None:
        clbl=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl)+'}$'
        cb.set_label(clbl, fontsize=20.)
    cb.ax.tick_params(labelsize=17)

    # Coast lines
    if not tricontour:
        ax.coastlines(zorder=10)
        ax.set_global()
    
        gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
            color='black', alpha=0.5, linestyle=':', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right=False
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'color': 'black'}# 'weight': 'bold'}
        gl.ylabel_style = {'color': 'black'}# 'weight': 'bold'}
        #gl.xlocator = mticker.FixedLocator([-180., -160, -140, -120, -60, -20.])
        #gl.xlocator = mticker.FixedLocator([-240., -180., -120, -65, -60, -55, 0, 60, 120.])
        #gl.ylocator = mticker.FixedLocator([0., 15., 30., 45, 60.])


    # Layout and save
    plt.savefig('dt_viirs_99',dpi=600)
    plt.show()
        

    return ax