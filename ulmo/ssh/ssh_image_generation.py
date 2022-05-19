import numpy as np
import pandas as pd
import seaborn as sns
import h5py
import matplotlib.pyplot as plt
import time
import os
from pkg_resources import resource_filename
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
#from ulmo.plotting import show_image

TimerStart = time.perf_counter()

dp_path = r"C:\Users\btpri\Desktop\SSH_std.parquet"
dp = pd.read_parquet(dp_path)
#print(dp.iloc[0])

dh_path = r"C:\Users\btpri\Desktop\SSH_100clear_32x32.h5"
dh = h5py.File(dh_path, 'r')
#print(list(dh.keys()))
valid = (dh['valid'])

# These two functions are copied from plotting.py to get around dependency issues
def load_palette(pfile=None):
    """ Load the color pallette

    Args:
        pfile (str, optional): Filename of the pallette. Defaults to None.

    Returns:
        color pallette, LinearSegmentedColormap: pallette for sns, colormap
    """
    
    if pfile is None:
        pfile = os.path.join(resource_filename('ulmo', 'plotting'), 'color_palette.txt')
    # Load me up
    with open(pfile, 'r') as f:
        colors = np.array([l.split() for l in f.readlines()]).astype(np.float32)
        pal = sns.color_palette(colors)
        boundaries = np.linspace(0, 1, 64)
        colors = list(zip(boundaries, colors))
        cm = LinearSegmentedColormap.from_list(name='rainbow', colors=colors)
    return pal, cm
def show_image(img:np.ndarray, cm=None, cbar=True, flipud=True,
               vmnx=(None,None), show=False, set_aspect=None):
    """Dispay the cutout image

    Args:
        img (np.ndarray): cutout image
        cm ([type], optional): Color map to use. Defaults to None.
            If None, load the heatmap above
        cbar (bool, optional): If True, show a color bar. Defaults to True.
        flipud (bool, optional): If True, flip the image up/down. Defaults to True.
        vmnx (tuple, optional): Set vmin, vmax. Defaults to None
        set_aspect (str, optional):
            Passed to ax.set_aspect() if provided

    Returns:
        matplotlib.Axis: axis containing the plot
    """
    if cm is None:
        _, cm = load_palette()
    #
    ax = sns.heatmap(np.flipud(img), xticklabels=[], 
                     vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap=cm, cbar=cbar)
    if show:
        plt.show()
    if set_aspect is not None:
        ax.set_aspect(set_aspect)
    #
    return ax

# For Creating the LL Histogram 
def LL_hist(dp=dp,logy=False):
    '''Makes the LL plot. If logy is set to true it will log the y-axis. Set to False by default.'''
    dp_plot = sns.histplot(dp, x="LL")
    
    if logy ==True:
        dp_plot.set_yscale('log')
    
# Shortens the .h5 file
def shorten_h5(ds, n, stats=False):
        """Shortens an array, ds, based on the first index to the length n.
        If stats is set to True it will print additional information; stats is False by default."""
        if stats == True:
            print('preshorten ', ds)
            print('preshorten length ', len(ds))
            print('preshorten shape ',ds.shape)
            print('')
            
        r = np.random.choice(len(ds), n)
        rs = np.sort(r)
        ds = ds[rs]
        
        if stats == True:
            print('short length ', len(ds))
            print('short shape ',ds.shape)
            print('')
            
        return ds

# Shortens the .parquet file
def shorten_parquet(num, ds, stats=False):
        """Shortens an array, ds, based on the first index to the length n.
        If stats is set to True it will print additional information; stats is False by default."""
        if stats == True:
            #print('preshorten ', ds)
            print('preshorten length ', len(ds))
            print('preshorten shape ',ds.shape)
            print('')
            
        ds = dp.sample(n=num)
        
        if stats == True:
            print('short length ', len(ds))
            print('short shape ',ds.shape)
            print('')
            
        return ds

# Plots subfiles and distinguishes extremes
def extreme_subfields(ds, points, tolerance):
    ''' Plots subfield positions and distinguishes the extremes. Takes data, 
    the amount of points to plot and a tolerance level based on the LL histogram.'''
    ds = shorten_parquet(points, ds)
    
    lats = ds.lat    
    lons = ds.lon

    extremes = np.where(ds.LL < tolerance)
    extremes = ds.iloc[extremes]

    exlats = extremes.lat
    exlons = extremes.lon

    # Fixing the coordinates so it displays all the points
    for i in range(len(lons)):
        if lons.iloc[i] >= 180:
            lons.iloc[i] = lons.iloc[i]-360

    for i in range(len(exlons)):
        if exlons.iloc[i] >= 180:
            exlons.iloc[i] = exlons.iloc[i]-360
    

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    fig.set_size_inches([8,8])
    ax.set_title("32x32 pixel grid {} sub fields points".format(points), size=17)
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.plot(lons, lats, '.', markersize = 1)
    ax.plot(exlons, exlats, '.', markersize = 2, color = "r", label="LL < {}".format(tolerance))
    ax.legend()

###### For some reason the plots only work correctly if done one at a time. ###########
extreme_subfields(dp,7841887,-100)

#LL_hist(logy=(True))

#valid = shorten_h5(valid,100)
#show_image(valid[12345][0])


TimerStop = time.perf_counter() 
RunTime = TimerStop - TimerStart
print(("total run time: {:.2f} seconds").format(RunTime))
