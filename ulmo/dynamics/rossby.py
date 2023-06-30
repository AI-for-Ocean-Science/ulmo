""" Methods related to the Rossby radius of deformation """

import os
from pkg_resources import resource_filename
import numpy as np

from astropy.coordinates import SkyCoord, match_coordinates_sky

import pandas

from IPython import embed

def load_rossdata():
    ross_file = os.path.join(
            resource_filename('ulmo', 'data'),
            'Rossby', 'rossrad.dat')

    ross = pandas.read_csv(ross_file, delim_whitespace=True,
                           comment='#')
    return ross

def calc_rossby_radius(lon:np.ndarray, lat:np.ndarray):
    # Load Rossby data
    ross = load_rossdata()
    ross_coord = SkyCoord(ra=ross['lon'], dec=ross['lat'], unit='deg') 

    # Coords
    in_coord = SkyCoord(ra=lon, dec=lat, unit='deg')

    idx, sep2d, _ = match_coordinates_sky(
        in_coord, ross_coord, nthneighbor=1)

    # Fill
    rossby_radius = ross['r'].values[idx]

    # Return
    return rossby_radius


# testing
if __name__ == '__main__':
    ross = load_rossdata()

    lons = np.linspace(0., 360., 100)
    lats = np.linspace(0., 40., 100)
    dum_coord = SkyCoord(ra=lons, dec=lats, unit='deg')

    r = calc_rossby_radius(lons, lats)