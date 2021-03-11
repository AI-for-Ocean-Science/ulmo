""" Basic I/O routines for the LLC analysis """

import os

import xarray as xr


def load_coords(verbose=True):
    coord_file = os.path.join(os.getenv('LLC_DATA'), 'LLC_coords.nc')
    if verbose:
        print("Loading LLC coords from {}".format(coord_file))
    coord_ds = xr.load_dataset(coord_file)
    return coord_ds

def load_CC_mask(field_size=(64,64), verbose=True):
    CC_mask_file = os.path.join(os.getenv('LLC_DATA'), 
                                'LLC_CC_mask_{}.nc'.format(field_size[0]))
    if verbose:
        print("Loading LLC CC mask from {}".format(CC_mask_file))
    CC_mask = xr.load_dataset(CC_mask_file)
    return CC_mask