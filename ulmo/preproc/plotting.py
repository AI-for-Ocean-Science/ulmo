""" Plotting routines related to extraction and pre-proc """
import numpy as np
import pandas

from matplotlib import pyplot as plt

try:
    import cartopy.crs as ccrs
except ImportError:
    print("cartopy not installed..")

# Astronomy tools
import astropy_healpix
from astropy import units


def plot_extraction(llc_table:pandas.DataFrame, figsize=(7,4),
                    resol=None, cbar=False, s=0.01):
    """Plot the extractions to check

    Args:
        llc_table (pandas.DataFrame): table of cutouts
        figsize (tuple, optional): Sets the figure size
        resol (float, optional): Angle in deg for healpix check. Defaults to None.
        cbar (bool, optional): [description]. Defaults to False.
        s (float, optional): [description]. Defaults to 0.01.
    """

    fig = plt.figure(figsize=figsize)
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
        nside = astropy_healpix.pixel_resolution_to_nside(resol*units.deg)
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
