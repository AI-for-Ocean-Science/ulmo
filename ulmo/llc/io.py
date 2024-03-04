""" Basic I/O routines for the LLC analysis """

import os
import warnings

import numpy as np
import xarray as xr
import pandas


from ulmo import io as ulmo_io
from ulmo.utils import image_utils

from IPython import embed

s3_llc_files_path = 's3://llc/ThetaUVSalt'

def load_coords(verbose=True):
    """Load LLC coordinates

    Args:
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        xarray.DataSet: contains the LLC coordinates
    """
    coord_file = os.path.join(os.getenv('OS_OGCM'), 
                              'LLC', 'data', 'LLC_coords.nc')
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
        CC_mask_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'data', 'CC',
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
                    
def load_llc_ds(filename, local=False):
    """
    Args:
        filename: (str) path of the file to be read.
        local: (bool) flag to show if the file is local or not.
    Returns:
        ds: (xarray.Dataset) Dataset.
    """
    if not local:
        with ulmo_io.open(filename, 'rb') as f:
            ds = xr.open_dataset(f)
    else:
        ds = xr.open_dataset(filename)
    return ds

def grab_cutout(data_var, row, col, field_size=None, fixed_km=None,
                coords_ds=None, resize=False):
    if field_size is None and fixed_km is None:
        raise IOError("Must set field_size or fixed_km")
    if coords_ds is None:
        coords_ds = load_coords()
    # Setup
    R_earth = 6371. # km
    circum = 2 * np.pi* R_earth
    km_deg = circum / 360.

    if fixed_km is not None:
        dlat_km = (coords_ds.lat.data[row+1,col]-coords_ds.lat.data[row,col]) * km_deg
        dr = int(np.round(fixed_km / dlat_km))
    else:
        dr = field_size
    dc = dr

    cut_data = data_var[row:row+dr, col:col+dc]

    if resize:
        raise NotImplementedError("Need to resize..")

    # Return
    return cut_data
                    
def grab_image(args):
    warnings.warn('Use grab_image() in utils.image_utils',
                  DeprecationWarning)
    return image_utils.grab_image(args)


def grab_velocity(cutout:pandas.core.series.Series, ds=None,
                  add_SST=False, add_Salt:bool=False, 
                  add_W=False, 
                  local_path:str=None):
    """Grab velocity

    Args:
        cutout (pandas.core.series.Series): cutout image
        ds (xarray.DataSet, optional): Dataset. Defaults to None.
        add_SST (bool, optional): Include SST too?. Defaults to False.
        add_Salt (bool, optional): Include Salt too?. Defaults to False.
        add_W (bool, optional): Include wz too?. Defaults to False.
        local_path (str, optional): Local path to data. Defaults to None.

    Returns:
        list: U, V cutouts as np.ndarray (i.e. values)
            and SST too if add_SST=True
            and Salt too if add_Salt=True
            and W too if add_W=True
    """
    # Local?with ulmo_io.open(cutout.filename, 'rb') as f:
    if local_path is None:
        filename = cutout.filename
    else:
        filename = os.path.join(local_path, os.path.basename(cutout.filename))
    # Open
    ds = xr.open_dataset(filename)

    # U field
    U_cutout = ds.U[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size].values
    # Vfield
    V_cutout = ds.V[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size].values
    output = [U_cutout, V_cutout]

    # Add SST?
    if add_SST:
        output.append(ds.Theta[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size].values)

    # Add Salt?
    if add_Salt:
        output.append(ds.Salt[cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size].values)

    # Add W
    if add_W:
        output.append(ds.W[0, cutout.row:cutout.row+cutout.field_size, 
                cutout.col:cutout.col+cutout.field_size].values)

    # Return
    return output
