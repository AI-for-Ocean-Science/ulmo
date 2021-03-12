""" Module for uniform analyses of LLC outputs"""

import os
import glob
import numpy as np

import pandas

import xarray as xr
import h5py

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import seaborn as sns

from ulmo.preproc import utils as pp_utils
from ulmo.preproc import extract as pp_extract
from ulmo.llc import io as llc_io

# Astronomy tools
import astropy_healpix
from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky


from IPython import embed

def add_days(llc_table, dti, outfile=None):
    # Check
    if 'datetime' in llc_table.keys():
        print("Dates already specified.  Not modifying")
        return llc_table

    # Do it
    llc_table['datetime'] = dti[0]
    for date in dti[1:]:
        new_tbl = llc_table[llc_table['datetime'] == dti[0]].copy()
        new_tbl['datetime'] = date
        llc_table = llc_table.append(new_tbl, ignore_index=True)

    # Write
    if outfile is not None:
        llc_io.write_llc_table(llc_table, outfile)

    # Return
    return llc_table

def build_CC_mask(filename=None, temp_bounds=(-3, 34), 
                  field_size=(64,64)):

    if filename is None:
        filename = os.path.join(os.getenv('LLC_DATA'), 
                                'ThetaUVSalt',
                                'LLC4320_2011-09-13T00_00_00.nc')
        
    #sst, qual = ulmo_io.load_nc(filename, verbose=False)
    print("Using file = {} for the mask".format(filename))
    ds = xr.load_dataset(filename)
    sst = ds.Theta.values

    # Generate the masks
    mask = pp_utils.build_mask(sst, None, temp_bounds=temp_bounds)
    
    # Sum across the image
    CC_mask = pp_extract.uniform_filter(mask.astype(float), 
                                        field_size[0], 
                                        mode='constant', 
                                        cval=1.)

    # Clear
    mask_edge = np.zeros_like(mask)
    mask_edge[:field_size[0]//2,:] = True
    mask_edge[-field_size[0]//2:,:] = True
    mask_edge[:,-field_size[0]//2:] = True
    mask_edge[:,:field_size[0]//2] = True

    #clear = (CC_mask < CC_max) & np.invert(mask_edge)
    CC_mask[mask_edge] = 1.  # Masked

    # Return
    return CC_mask

def uniform_coords(resol, field_size, CC_max=1e-4, outfile=None):
    """
    Use healpix to setup a uniform extraction grid
    Args:
        resol (float): Typical separation on the healpix grid
        field_size (tuple): Cutout size in pixels
        outfile (str, optional): If provided, write the table to this outfile.
            Defaults to None.

    Returns:
        pandas.DataFrame: Table containing the coords
    """
    # Load up CC_mask
    CC_mask = llc_io.load_CC_mask(field_size=field_size)
    # Cut
    good_CC = CC_mask.CC_mask.values < CC_max
    good_CC_idx = np.where(good_CC)

    # Build coords
    llc_lon = CC_mask.lon.values[good_CC].flatten()
    llc_lat = CC_mask.lat.values[good_CC].flatten()
    print("Building LLC SkyCoord")
    llc_coord = SkyCoord(llc_lon*units.deg + 180.*units.deg, 
                         llc_lat*units.deg, 
                         frame='galactic')

    # Healpix time
    nside = astropy_healpix.pixel_resolution_to_nside(resol*units.deg)
    hp = astropy_healpix.HEALPix(nside=nside)
    hp_lon, hp_lat = hp.healpix_to_lonlat(np.arange(hp.npix))

    # Coords
    hp_coord = SkyCoord(hp_lon, hp_lat, frame='galactic')
                        
    # Cross-match
    print("Cross-match")
    idx, sep2d, _ = match_coordinates_sky(hp_coord, llc_coord, nthneighbor=1)
    flag = np.zeros(len(llc_coord), dtype='bool')
    good_sep = sep2d < hp.pixel_resolution
    good_sep_idx = np.where(good_sep)

    # Build the table
    llc_table = pandas.DataFrame()
    llc_table['lat'] = llc_lat[idx[good_sep]]  # Center of cutout
    llc_table['lon'] = llc_lon[idx[good_sep]]  # Center of cutout

    llc_table['row'] = good_CC_idx[0][idx[good_sep]] - field_size[0]//2 # Lower left corner
    llc_table['col'] = good_CC_idx[1][idx[good_sep]] - field_size[0]//2 # Lower left corner
    
    if outfile is not None:
        llc_io.write_llc_table(llc_table, outfile)
    return llc_table


def plot_extraction(llc_table, resol=None, cbar=False, s=0.01):

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

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Build CC_mask for 64x64
    if flg & (2**0):  
        CC_mask = build_CC_mask(field_size=(64,64))
        # Load coords
        coord_ds = llc_io.load_coords()
        # New dataset
        ds = xr.Dataset(data_vars=dict(CC_mask=(['x','y'], CC_mask)),
                                       coords=dict(
                                           lon=(['x','y'], coord_ds.lon),
                                           lat=(['x','y'], coord_ds.lat)))
        # Write
        filename = os.path.join(os.getenv('LLC_DATA'), 
                                'LLC_CC_mask_64.nc')
        ds.to_netcdf(filename)
        print("Wrote: {}".format(filename))


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # Build CC mask
    else:
        flg = sys.argv[1]

    main(flg)