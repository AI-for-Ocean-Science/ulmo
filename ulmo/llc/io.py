""" Basic I/O routines for the LLC analysis """

import os

import xarray as xr
import pandas
import h5py

from ulmo import io as ulmo_io

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
        with ulmo_io.open(CC_mask_file, 'rb') as f:
            CC_mask = xr.open_dataset(f)
    if verbose:
        print("Loading LLC CC mask from {}".format(CC_mask_file))
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
                    
                    
def grab_image(cutout:pandas.core.series.Series, 
               close=True, pp_hf=None):                
    """Grab the cutout image

    Args:
        cutout (pandas.core.series.Series): [description]
        close (bool, optional): [description]. Defaults to True.
        pp_hf ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    # Open?
    if pp_hf is None:
        with ulmo_io.open(cutout.pp_file, 'rb') as f:
            pp_hf = h5py.File(f, 'r')
    img = pp_hf['valid'][cutout.pp_idx, 0, ...]

    # Close?
    if close:
        pp_hf.close()
        return img
    else:
        return img, pp_hf