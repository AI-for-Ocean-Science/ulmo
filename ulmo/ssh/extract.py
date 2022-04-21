""" Extraction routines for VIIRS """

import os
import numpy as np

from ulmo.ssh import io as ssh_io
from ulmo.preproc import utils as pp_utils
from ulmo.preproc import extract

from IPython import embed


def extract_file(filename: str,
                 field_size=(192, 192),
                 nadir_offset=0,
                 CC_max=0.05,
                 qual_thresh=5,
                 temp_bounds=(-3, 34),
                 nrepeat=1,
                 sub_grid_step=2,
                 lower_qual=False,
                 inpaint=False, debug=False):
    """Method to extract a single file.
    Usually used in parallel

    This is very similar to the MODIS routine

    Args:
        filename (str): VIIRS datafile with path
        field_size (tuple, optional): [description]. Defaults to (128,128).
        nadir_offset (int, optional): [description]. Defaults to 480.
            Zero means none.
        CC_max (float, optional): [description]. Defaults to 0.05.
        qual_thresh (int, optional): [description]. Defaults to 2.
        lower_qual (bool, optional): 
            If False, threshold is an upper bound for masking
        temp_bounds (tuple, optional): [description]. Defaults to (-2, 33).
        nrepeat (int, optional): [description]. Defaults to 1.
        sub_grid_step (int, optional):  Sets how finely to sample the image.
            Larger means more finely
        inpaint (bool, optional): [description]. Defaults to False.
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        tuple: raw_ssh, inpainted_mask, metadata
    """

    # Load the image

    ssh, latitude, longitude = ssh_io.load_nc(filename, verbose=True)
    if ssh is None:
        return

    # Generate the masks
    #masks = pp_utils.build_mask(ssh, qual, 
    #                            qual_thresh=qual_thresh,
    #                            temp_bounds=temp_bounds, 
    #                            lower_qual=lower_qual)
    masks = np.ones_like(ssh)
    bad = np.isnan(ssh)
    masks[bad] = 0




    ssh, latitude, longitude = ssh_io.load_nc(filename, verbose=True)
    
    if ssh is None:
        return

    # Generate the dummy masks
    ones = np.ones_like(ssh)    
    masks = ones == 0



    # Restrict to near nadir
    nadir_pix = ssh.shape[1] // 2
    if nadir_offset > 0:
        lb = nadir_pix - nadir_offset
        ub = nadir_pix + nadir_offset
        ssh = ssh[:, lb:ub]
        masks = masks[:, lb:ub].astype(np.uint8)
    else:
        lb = 0

    # Random clear rows, cols
    rows, cols, clear_fracs = extract.clear_grid(
        masks, field_size[0], 'center',
        CC_max=CC_max, nsgrid_draw=nrepeat,
        sub_grid_step=sub_grid_step)
    
    if rows is None:
        return None

    
    # Extract
    fields, inpainted_masks = [], []
    metadata = []
    for r, c, clear_frac in zip(rows, cols, clear_fracs):
        # Inpaint?
        field = ssh[r:r+field_size[0], c:c+field_size[1]]
        mask = masks[r:r+field_size[0], c:c+field_size[1]]
        if inpaint:
            inpainted, _ = pp_utils.preproc_field(
                field, mask, only_inpaint=True)
        if inpainted is None:
            continue
        # Null out the non inpainted (to preseve memory when compressed)
        inpainted[~mask] = np.nan
        # Append ssh raw + inpainted
        fields.append(field.astype(np.float32))
        inpainted_masks.append(inpainted)
        # meta
        row, col = r, c + lb
        lat = latitude[col + field_size[0] // 2]#, col + field_size[1] // 2]
        lon = longitude[row + field_size[0] // 2]#, col + field_size[1] // 2]
        metadata.append([filename, str(row), str(
            col), str(lat), str(lon), str(clear_frac)])

    del ssh, masks

    return np.stack(fields), np.stack(inpainted_masks), np.stack(metadata)


#fn = "https://opendap.jpl.nasa.gov/opendap/SeaSurfaceTopography/merged_alt/L4/cdr_grid/ssh_grids_v1812_1992100212.nc"
#test = extract_file(fn)
#print(test[1])


