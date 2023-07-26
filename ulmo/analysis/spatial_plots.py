# create spatial distribution plots! 

#imports
import pandas
import healpy as hp
import numpy as np
from matplotlib import pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from ulmo.utils import image_utils


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

def show_avg_LL(main_tbl:pandas.DataFrame, 
                 nside=64, 
                 use_mask=True, tricontour=False,
                 lbl=None, figsize=(12,8), 
                 color='viridis', show=True):
    """Generate a global map of mean LL of the input
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
    hp_events, hp_lons, hp_lats, hp_values = evals_to_healpix(
        main_tbl, nside, mask=use_mask)
    
    # Figure
    
    fig = plt.figure(figsize=figsize)
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    if tricontour:
        cm = plt.get_cmap(color)
        img = ax.tricontourf(hp_lons, hp_lats, hp_values, transform=tformM,
                         levels=20, cmap=cm)#, zorder=10)
    else:
        cm = plt.get_cmap(color)
        # Cut
        good = np.invert(hp_values.mask)
        img = plt.scatter(x=hp_lons[good],
            y=hp_lats[good],
            c=hp_values[good], vmin = -1000, vmax = 500, 
            cmap=cm,
            s=1,
            transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    if lbl is not None:
        clbl = 'mean_LL'
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
    if show:
        plt.show()

    return ax





def evals_to_healpix_stat(eval_tbl, nside,  mask=True,
                          metric:str='LL', stat:str='median'):
    """
    Find out where the input cutouts are located and the median values associated
    with each pixel; default is LL

    Parameters
    ----------
    eval_tbl : pandas.DataFrame
    nside : int  # nside is a number that sets the resolution of map
    mask : bool, optional
        If True, include a mask on the arrays
    stat : str, optional
        If 'median', return the median value of the metric in each pixel
        If 'mean', return the average value of the metric in each pixel
        else: barf

    Returns
    -------
    num of events, lats, lons, mean/median values : hp.ma, np.ndarray, np.ndarray, hp.ma

    """
    if stat not in ['mean', 'median']:
        raise IOError("Bad stat input")
    
    # Grab lats, lons
    lats = eval_tbl.lat.values
    lons = eval_tbl.lon.values

    # Values
    vals = eval_tbl[metric].values

    # Healpix coords
    theta = (90 - lats) * np.pi / 180.  # convert into radians
    phi = lons * np.pi / 180.
    idx_all = hp.pixelfunc.ang2pix(nside, theta, phi) # returns the healpix pixel numbers that correspond to theta and phi values

    # Intialize the arrays
    npix_hp = hp.nside2npix(nside)  # returns the number of pixels on map, based on nside parameter
    all_events = np.ma.masked_array(np.zeros(npix_hp, dtype='int')) # array of all pixels on map
    med_values = np.ma.masked_array(np.zeros(npix_hp, dtype='float')) # will contain median LL value in that pixel

    # Count events
    for i, idx in enumerate(idx_all):
        all_events[idx] += 1 # pixels concentrated with data pts >= 1 ; those without data remain 0

    zero = all_events == 0 
    float_events = all_events.astype(float)
# ~ operator is called the complement bitwise operator 
# inverts the True/False values
# [~zero] selects pixels where the cutouts are (where events = 1 exist)


    # Calculate median values
    #idx_arr = pandas.Series(idx_all).sort_values()
    #pixels = pandas.unique(idx_arr)

    pixels = np.unique(idx_all)

    for pixel in pixels: 
    
        # find where which cutouts to put in that pixel
        #where = np.where(pixel == idx_arr)
        #first = where[0][0]
        #last = where[0][-1]
        #indices = idx_arr[first:last + 1].index

        good = pixel == idx_all
    
        # evaluate the median value for that pixel 
        #sub_vals = eval_tbl.iloc[indices.to_numpy()][metric].to_numpy()
        sub_vals = vals[good]
    
        if stat == 'median':
            med_values[pixel] = np.median(sub_vals)
        elif stat == 'mean':
            med_values[pixel] = np.mean(sub_vals)
        #med_values[pixel] = np.median( vals )


    # Mask
    evts = hp.ma(float_events)
    meds = hp.ma(med_values)
    if mask:  # if you want to mask float_events
        evts.mask = zero # current mask set to zero array, where Trues (no events) are masked
        meds.mask = zero 

    # Angles
    hp_lons, hp_lats = hp.pixelfunc.pix2ang(nside, np.arange(npix_hp), lonlat=True)

    # Return
    return evts, hp_lons, hp_lats, meds



def show_med_LL(main_tbl:pandas.DataFrame, 
                 nside=64, 
                 use_mask=True, tricontour=False,
                 lbl=None, figsize=(12,8), 
                 metric='LL',
                 color='viridis', show=True):
    """Generate a global map of median LL of cutouts at that location 
    
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
        metric (str, optional): Metric to plot. Defaults to 'LL'.
    Returns:
        matplotlib.Axis: axis holding the plot
    """
    # Healpix me
    hp_events, hp_lons, hp_lats, hp_values = evals_to_healpix_stat(
        main_tbl, nside, mask=use_mask, metric=metric)
    
    # Figure
    
    fig = plt.figure(figsize=figsize)
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    if tricontour:
        cm = plt.get_cmap(color)
        img = ax.tricontourf(hp_lons, hp_lats, hp_values, transform=tformM,
                         levels=20, cmap=cm)#, zorder=10)
    else:
        cm = plt.get_cmap(color)
        # Cut
        good = np.invert(hp_values.mask)
        img = plt.scatter(x=hp_lons[good],
            y=hp_lats[good],
            c=hp_values[good], vmin = -1000, vmax = 500, 
            cmap=cm,
            s=1,
            transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    if lbl is not None:
        clbl = 'median LL'
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
    if show:
        plt.show()

    return ax

def show_spatial_two_avg(tbl1:pandas.DataFrame, tbl2:pandas.DataFrame, 
                 nside=64, use_log=True, 
                 use_mask=True, tricontour=False,
                 lbl=None, figsize=(12,8), 
                 color='coolwarm', show=True):
    """Generate a global map of difference in the mean LL 

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
    hp_events1, hp_lons1, hp_lats1, hp_values1 = evals_to_healpix(
        tbl1, nside, log=use_log, mask=use_mask)
    
    hp_events2, hp_lons2, hp_lats2, hp_values2 = evals_to_healpix(
        tbl2, nside, log=use_log, mask=use_mask)
    
    # Figure
    
    fig = plt.figure(figsize=figsize)
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    if tricontour:
        cm = plt.get_cmap(color)
        img = ax.tricontourf(hp_lons1, hp_lats1, hp_values1 - hp_values2, transform=tformM,
                         levels=20, cmap=cm)#, zorder=10)
    else:
        cm = plt.get_cmap(color)
        # Cut
        good = np.invert(hp_values2.mask)
        img = plt.scatter(x=hp_lons2[good],
            y=hp_lats2[good],
            c=hp_values1[good]- hp_values2[good], vmin = -300, vmax = 300, 
            cmap=cm,
            s=1,
            transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    if lbl is not None:
        
        clbl = 'diff_mean_LL'
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
    if show:
        plt.show()

    return ax



def show_spatial_two_med(tbl1:pandas.DataFrame, tbl2:pandas.DataFrame, 
                 nside=64, 
                 use_mask=True, tricontour=False,
                 lbl=None, figsize=(12,8), 
                 color='coolwarm', show=True):
    """Generate a global map of the difference in median LL

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
    hp_events1, hp_lons1, hp_lats1, hp_values1 = evals_to_healpix_meds(
        tbl1, nside, mask=use_mask)
    
    hp_events2, hp_lons2, hp_lats2, hp_values2 = evals_to_healpix_meds(
        tbl2, nside, mask=use_mask)
    
    # Figure
    
    fig = plt.figure(figsize=figsize)
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    if tricontour:
        cm = plt.get_cmap(color)
        img = ax.tricontourf(hp_lons1, hp_lats1, hp_values1 - hp_values2, transform=tformM,
                         levels=20, cmap=cm)#, zorder=10)
    else:
        cm = plt.get_cmap(color)
        # Cut
        good = np.invert(hp_values2.mask)
        img = plt.scatter(x=hp_lons2[good],
            y=hp_lats2[good],
            c=hp_values1[good]- hp_values2[good], vmin = -300, vmax = 300, 
            cmap=cm,
            s=1,
            transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    if lbl is not None:
        
        clbl = 'diff median LL'
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
    if show:
        plt.show()

    return ax



def scatter_diff_avg(tbl1:pandas.DataFrame, tbl2:pandas.DataFrame, 
                 nside=32, use_log=False, 
                 use_mask=True, tricontour=False,
                 lbl=None, figsize=(12,8), 
                 color='plasma', show=True):
    """Generate a scatter plot with the difference of mean LL between two
    dataframes (pixel-wise) as a function of the number of cutouts

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
    hp_events1, hp_lons1, hp_lats1, hp_values1 = evals_to_healpix(
        tbl1, nside, log=use_log, mask=use_mask)
    
    hp_events2, hp_lons2, hp_lats2, hp_values2 = evals_to_healpix(
        tbl2, nside, log=use_log, mask=use_mask)
    
    # Figure
    
    fig = plt.figure(figsize=figsize)
    plt.clf()
    cm = plt.get_cmap(color)
        
    # Cut
    good = np.invert(hp_values2.mask)
    ax = plt.scatter(x= hp_events2[good] , y = hp_values1[good]- hp_values2[good], s=1)

    # Axis Labels
    plt.xlabel('num of cutouts : table 2')
    plt.ylabel('diff_mean_LL')
    plt.ylim(-1000, 1000)


    # Layout and save
    if show:
        plt.show()

    return ax



def scatter_diff_med(tbl1:pandas.DataFrame, tbl2:pandas.DataFrame, 
                 nside=32,  
                 use_mask=True, tricontour=False,
                 lbl=None, figsize=(12,8), 
                 color='plasma', show=True):
    """Generate a scatter plot with the difference of median LL between two
    dataframes (pixel-wise) as a function of the number of cutouts

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
    hp_events1, hp_lons1, hp_lats1, hp_values1 = evals_to_healpix_meds(
        tbl1, nside, mask=use_mask)
    
    hp_events2, hp_lons2, hp_lats2, hp_values2 = evals_to_healpix_meds(
        tbl2, nside, mask=use_mask)
    
    # Figure
    
    fig = plt.figure(figsize=figsize)
    plt.clf()
    cm = plt.get_cmap(color)
        
    # Cut
    good = np.invert(hp_values2.mask)
    ax = plt.scatter(x= hp_events2[good] , y = hp_values1[good]- hp_values2[good], s=1)

    # Axis Labels
    plt.xlabel('num of cutouts : table 2')
    plt.ylabel('diff_median_LL')
    plt.ylim(-1000, 1000)


    # Layout and save
    if show:
        plt.show()

    return ax


def show_spatial_two_slices(sub_tbl1:pandas.DataFrame, sub_tbl2:pandas.DataFrame,
                 nside=64, use_log=True, 
                 use_mask=True,
                 lbl1=None, lbl2=None, figsize=(24,16), 
                 color1='Reds', color2='Blues', show=True):
    """Show where most cutouts come from (from table 1 or table 2) on a global map 

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

    # Colorbar
    cb1 = plt.colorbar(img1, orientation='vertical', location = 'left', shrink = 0.25)
    cb2 = plt.colorbar(img2, orientation='vertical', location = 'right',shrink = 0.25)
    
    if lbl1 is not None:
        clbl1=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl1)+'}$'
        cb1.set_label(clbl1, fontsize=20.)
    cb1.ax.tick_params(labelsize=17)
    
    if lbl2 is not None:
        clbl2=r'$\log_{10} \, N_{\rm '+'{}'.format(lbl2)+'}$'
        cb2.set_label(clbl2, fontsize=20.)
    cb2.ax.tick_params(labelsize=17)

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
    if show:
        plt.show()

    return ax


def show_spatial_diff(sub_tbl1:pandas.DataFrame, sub_tbl2:pandas.DataFrame,
                 nside=64, use_log=True, 
                 use_mask=True,
                 lbl1=None, lbl2=None, lbl3=None, figsize=(24,16), 
                 color1='Reds', color2='Blues', color3='Greys', show=True):
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
    cb1 = plt.colorbar(img1, orientation='vertical', location = 'left', shrink = 0.25)
    cb2 = plt.colorbar(img2, orientation='vertical', location = 'right',shrink = 0.25)
    cb3 = plt.colorbar(img3, orientation='horizontal', location = 'bottom',shrink = 0.25)
    
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
    if show:
        plt.show()

    return ax



def evals_to_healpix1(eval_tbl, nside, mask=True):
    """
    Generate a healpix map of where the input
    MHW Systems are located on the globe, simplified
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
    """
    # Grab lats, lons
    lats = eval_tbl.lat.values
    lons = eval_tbl.lon.values

    # Healpix coords
    theta = (90 - lats) * np.pi / 180. 
    phi = lons * np.pi / 180.
    idx_all = hp.pixelfunc.ang2pix(nside, theta, phi) 

    # Count events
    npix_hp = hp.nside2npix(nside)
    all_events = np.ma.masked_array(np.zeros(npix_hp, dtype='int'))  

    for i, idx in enumerate(idx_all):
        all_events[idx] += 1

    zero = all_events == 0 
    
    float_events = all_events.astype(float)


    # Mask
    hpma = hp.ma(float_events)
    if mask:  # if you want to mask float_events
        hpma.mask = zero # current mask set to zero array, where Trues (no events) are masked

    # Angles
    hp_lons, hp_lats = hp.pixelfunc.pix2ang(nside, np.arange(npix_hp), lonlat=True)

    # Return
    return hpma, hp_lons, hp_lats



def show_spatial_diff1(sub_tbl1:pandas.DataFrame, sub_tbl2:pandas.DataFrame,
                 nside=64, use_log=True, 
                 use_mask=True,
                 lbl1=None, lbl2=None, figsize=(24,16), 
                 color ='coolwarm', show=True):
    """Generate a global map of the location of the input
    cutouts and what their source is.
    
    Table 1 cutouts are blue
    Table 2 cutouts are red
    
    Shared pixels are colored in a range from blue to gray to red to what the main source is.

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
    hp_events1, hp_lons1, hp_lats1= evals_to_healpix1(
        sub_tbl1, nside, mask=use_mask)
    
    hp_events2, hp_lons2, hp_lats2= evals_to_healpix1(
        sub_tbl2, nside, mask=use_mask)
    
    # Figure
    
    fig = plt.figure(figsize=figsize)
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)

    cm = plt.get_cmap(color)
    
    # Masks
    
    
    a = (hp_events1.mask == False)
    b = (hp_events2.mask == False)
    both = a & (a == b)
    
    
    good1 = a & ( ~both)
    good2 = b & ( ~both)
    
    
    # Cut
    img1 = plt.scatter(x=hp_lons1[good1],
        y=hp_lats1[good1],
        c= cm(0),
        s=5,
        transform=tformP)
    
    img2 = plt.scatter(x=hp_lons2[good2],
        y=hp_lats2[good2],
        c=cm(1),
        s=5,
        transform=tformP)
    
    img3 = plt.scatter(x=hp_lons2[both],
        y=hp_lats2[both],
        c=hp_events1[both] - hp_events2[both], vmin = -500, vmax = 500,
        cmap=cm,
        s=5,
        transform=tformP) 
    #c=hp_events2[both]

    # Colorbar
    cb = plt.colorbar(img3, orientation='vertical', location = 'right',shrink = 0.25)
    
    if lbl1 is not None and lbl2 is not None:
        clbl='{} - {}'.format(lbl1, lbl2)
        cb.set_label(clbl, fontsize=20.)
    cb.ax.tick_params(labelsize=17)
    

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
    if show:
        plt.show()

    return ax