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

def grab_image(cutout:pandas.core.series.Series, 
               close=True, pp_hf=None):                
    """Grab the pre-processed image of a cutout

    Args:
        cutout (pandas.core.series.Series): cutout
        close (bool, optional): If True, close the file afterwards. Defaults to True.
        pp_hf ([type], optional): [description]. Defaults to None.

    Returns:
        np.ndarray: Image of the cutout
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
