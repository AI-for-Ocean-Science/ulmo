""" Extraction routines for MODIS """

import os
import numpy as np


from ulmo.preproc import utils as pp_utils
from ulmo.preproc import extract
from ulmo.modis import io as modis_io


from IPython import embed

def extract_file(filename:str, 
                 field='SST',
                 field_size=(128,128),
                 nadir_offset=480,
                 CC_max=0.05, 
                 qual_thresh=2,
                 temp_bounds = (-2, 33),
                 nrepeat=1,
                 inpaint=True, debug=False):
    """Method to extract a single file.
    Usually used in parallel

    Args:
        ifile (str): MODIS datafile
        field_size (tuple, optional): [description]. Defaults to (128,128).
        nadir_offset (int, optional): [description]. Defaults to 480.
        CC_max (float, optional): [description]. Defaults to 0.05.
        qual_thresh (int, optional): [description]. Defaults to 2.
        temp_bounds (tuple, optional): [description]. Defaults to (-2, 33).
        nrepeat (int, optional): [description]. Defaults to 1.
        inpaint (bool, optional): [description]. Defaults to True.
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        tuple: fields, field_masks, metadata
    """
    # Load up
    sst, latitude, longitude, masks = modis_io.load_granule(
        filename, field=field, qual_thresh=qual_thresh,
        temp_bounds=temp_bounds)
    if sst is None:
        return

    # Restrict to near nadir
    nadir_pix = sst.shape[1] // 2
    lb = nadir_pix - nadir_offset
    ub = nadir_pix + nadir_offset
    sst = sst[:, lb:ub]
    masks = masks[:, lb:ub].astype(np.uint8)

    # Random clear rows, cols
    rows, cols, clear_fracs = extract.clear_grid(
        masks, field_size[0], 'center', 
        CC_max=CC_max, nsgrid_draw=nrepeat)
    if rows is None:
        return

    # Extract
    fields, field_masks = [], []
    metadata = []
    for r, c, clear_frac in zip(rows, cols, clear_fracs):
        # Inpaint?
        field = sst[r:r+field_size[0], c:c+field_size[1]]
        mask = masks[r:r+field_size[0], c:c+field_size[1]]
        if inpaint:
            field, _ = pp_utils.preproc_field(field, mask, only_inpaint=True)
        if field is None:
            continue
        # Append SST and mask
        fields.append(field.astype(np.float32))
        field_masks.append(mask)
        # meta
        row, col = r, c + lb
        lat = latitude[row + field_size[0] // 2, col + field_size[1] // 2]
        lon = longitude[row + field_size[0] // 2, col + field_size[1] // 2]
        metadata.append([filename, str(row), str(col), str(lat), str(lon), str(clear_frac)])

    del sst, masks

    return np.stack(fields), np.stack(field_masks), np.stack(metadata)
