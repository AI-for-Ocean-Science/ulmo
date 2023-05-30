""" Module for uniform analyses of LLC outputs"""

import numpy as np


import pandas

from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io

# Astronomy tools
import astropy_healpix

from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky


from IPython import embed


def coords(resol, field_size, CC_max=1e-4, outfile=None, 
           max_lat=None, localCC=True, min_lat:float=None,
           rotate:float=None):
    """
    Use healpix to setup a uniform extraction grid

    Args:
        resol (float): Typical separation on the healpix grid (deg?)
        field_size (tuple): Cutout size in pixels
        max_lat (float,optional): Restrict to latitudes lower than this
        min_lat (float,optional): Restrict to latitudes higher than this
        outfile (str, optional): If provided, write the table to this outfile.
            Defaults to None.
        localCC (bool, optional):  If True, load the CC_mask locally.
        rotate (float, optional): Rotate the grid by this angle (deg)

    Returns:
        pandas.DataFrame: Table containing the coords
    """
    # Load up CC_mask
    CC_mask = llc_io.load_CC_mask(field_size=field_size, local=localCC)

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
    if rotate is not None:
        hp_lon = hp_lon + rotate*np.pi/180. * units.rad

    # Coords
    hp_coord = SkyCoord(hp_lon, hp_lat, frame='galactic')
                        
    # Cross-match
    print("Cross-match")
    idx, sep2d, _ = match_coordinates_sky(hp_coord, llc_coord, nthneighbor=1)
    good_sep = sep2d < hp.pixel_resolution

    # Build the table
    llc_table = pandas.DataFrame()
    llc_table['lat'] = llc_lat[idx[good_sep]]  # Center of cutout
    llc_table['lon'] = llc_lon[idx[good_sep]]  # Center of cutout

    llc_table['row'] = good_CC_idx[0][idx[good_sep]] - field_size[0]//2 # Lower left corner
    llc_table['col'] = good_CC_idx[1][idx[good_sep]] - field_size[0]//2 # Lower left corner

    # Cut on latitutde?
    if max_lat is not None:
        print(f"Restricting to |latitude| < {max_lat}")
        gd_lat = np.abs(llc_table.lat) < max_lat
        llc_table = llc_table[gd_lat].copy()
    
    if min_lat is not None:
        print(f"Restricting to |latitude| > {min_lat}")
        gd_lat = np.abs(llc_table.lat) > min_lat
        llc_table = llc_table[gd_lat].copy()

    llc_table.reset_index(inplace=True)
    
    # Write
    if outfile is not None:
        ulmo_io.write_main_table(llc_table, outfile)

    # Return
    return llc_table

