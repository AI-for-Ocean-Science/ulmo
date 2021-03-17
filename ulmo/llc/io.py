""" Basic I/O routines for the LLC analysis """

import os

import pandas
import xarray as xr

from IPython import embed

llc_files_path = os.path.join(os.getenv('LLC_DATA'), 'ThetaUVSalt')

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

def build_llc_datafile(date=None, root='LLC4320_', chk=True):
    if date is not None:
        sdate = str(date).replace(':','_')[:19]
        datafile = os.path.join(llc_files_path, root+sdate+'.nc')
    if chk:
        try:
            assert os.path.isfile(datafile)
        except:
            embed(header='34 of io')
    # Return
    return datafile
                    