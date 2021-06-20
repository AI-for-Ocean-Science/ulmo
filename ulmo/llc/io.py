""" Basic I/O routines for the LLC analysis """

import os
import warnings

import xarray as xr
import pandas
import h5py

from ulmo import io as ulmo_io
from ulmo.utils import image_utils

from IPython import embed

if os.getenv('LLC_DATA') is not None:
    local_llc_files_path = os.path.join(os.getenv('LLC_DATA'), 'ThetaUVSalt')
s3_llc_files_path = 's3://llc/ThetaUVSalt'

def load_coords(verbose=True):
    """Load LLC coordinates

    Args:
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        xarray.DataSet: contains the LLC coordinates
    """
    coord_file = os.path.join(os.getenv('LLC_DATA'), 'LLC_coords.nc')
    if verbose:
        print("Loading LLC coords from {}".format(coord_file))
    coord_ds = xr.load_dataset(coord_file)
    return coord_ds

def load_CC_mask(field_size=(64,64), verbose=True, local=True):
    """Load up a CC mask.  Typically used for setting coordinates

    Args:
        field_size (tuple, optional): Field size of the cutouts. Defaults to (64,64).
        verbose (bool, optional): Defaults to True.
        local (bool, optional): Load from local hard-drive. 
            Requires LLC_DATA env variable.  Defaults to True (these are 3Gb files)

    Returns:
        xr.DataSet: CC_mask
    """
    if local:
        CC_mask_file = os.path.join(os.getenv('LLC_DATA'), 'CC',
                                   'LLC_CC_mask_{}.nc'.format(field_size[0]))
        CC_mask = xr.open_dataset(CC_mask_file)
    else:
        CC_mask_file = 's3://llc/CC/'+'LLC_CC_mask_{}.nc'.format(field_size[0])
        CC_mask = xr.load_dataset(ulmo_io.load_to_bytes(CC_mask_file))
    if verbose:
        print("Loaded LLC CC mask from {}".format(CC_mask_file))
    # Return
    return CC_mask


def grab_llc_datafile(datetime=None, root='LLC4320_', chk=True, local=False):
    """Generate the LLC datafile name from the inputs

    Args:
        datetime (pandas.TimeStamp, optional): Date. Defaults to None.
        root (str, optional): [description]. Defaults to 'LLC4320_'.
        chk (bool, optional): [description]. Defaults to True.
        local (bool, optional): [description]. Defaults to False.

    Returns:
        str: LLC datafile name
    """
    # Path
    llc_files_path = local_llc_files_path if local else s3_llc_files_path
        
    if datetime is not None:
        sdate = str(datetime).replace(':','_')[:19]
        # Add T?
        if sdate[10] == ' ':
            sdate = sdate.replace(' ', 'T')
        # Finish
        datafile = os.path.join(llc_files_path, root+sdate+'.nc')
    if chk and local:
        try:
            assert os.path.isfile(datafile)
        except:
            embed(header='34 of io')
    # Return
    return datafile
                    
                    
def grab_image(**args):
    warnings.warn('Use grab_image() in utils.image_utils',
                  DeprecationWarning)
    return image_utils.grab_image(args)


def grab_velocity(cutout:pandas.core.series.Series, ds=None,
                  add_SST=False):                
    with ulmo_io.open(cutout.filename, 'rb') as f:
        ds = xr.open_dataset(f)
    # U field
    U_cutout = ds.U[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size]
    # Vfield
    V_cutout = ds.V[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size]
    output = [U_cutout, V_cutout]
    # Add SST?
    if add_SST:
        output.append(ds.Theta[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size])
    # Return
    return output