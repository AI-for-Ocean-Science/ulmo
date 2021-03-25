""" Plotting routines"""
# Hiding cartopy

import numpy as np
import pandas

from matplotlib import pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Astronomy tools
import astropy_healpix
from astropy import units

from ulmo.plotting import plotting
from ulmo.llc import io as llc_io

def plot_extraction(llc_table:pandas.DataFrame, 
                    resol=None, cbar=False, s=0.01):
    """Plot the extractions to check

    Args:
        llc_table (pandas.DataFrame): table of cutouts
        resol (float, optional): Angle in deg for healpix check. Defaults to None.
        cbar (bool, optional): [description]. Defaults to False.
        s (float, optional): [description]. Defaults to 0.01.
    """

    fig = plt.figure(figsize=(7, 4))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformM)


    cm = plt.get_cmap('Blues')
    # Cut
    #good = np.invert(hp_events.mask)
    img = plt.scatter(x=llc_table.lon,
        y=llc_table.lat,
        s=s,
        transform=tformP)

    # Healpix?
    if resol is not None:
        nside = astropy_healpix.pixel_resolution_to_nside(0.5*units.deg)
        hp = astropy_healpix.HEALPix(nside=nside)
        hp_lon, hp_lat = hp.healpix_to_lonlat(np.arange(hp.npix))
        img = plt.scatter(x=hp_lon.to('deg').value,
            y=hp_lat.to('deg').value,
            s=s,
            color='r',
            transform=tformP)

    #
    # Colorbar
    if cbar:
        cb = plt.colorbar(img, orientation='horizontal', pad=0.)
        cb.ax.tick_params(labelsize=17)

    # Coast lines
    ax.coastlines(zorder=10)
    ax.set_global()

    plt.show()

    return


def show_cutout(cutout:pandas.core.series.Series): 
    """Simple wrapper for showing the input cutout

    Args:
        cutout (pandas.core.series.Series): Cutout to display
    """


    # Load image
    img = llc_io.grab_image(cutout, close=True)

    # Plot
    plotting.show_cutout(img)
