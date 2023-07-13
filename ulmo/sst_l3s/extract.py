""" Module for extracting L3S data """

import os
import glob
import numpy as np

import pandas

import xarray as xr
import h5py


from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo import io as ulmo_io

from ulmo.preproc import utils as pp_utils
from ulmo.preproc import extract as pp_extract
from ulmo.preproc import io as pp_io
from ulmo.utils import catalog

from ulmo.llc import io as llc_io
from ulmo.llc import kinematics


from IPython import embed

def preproc_for_analysis(l3s_table:pandas.DataFrame, 
                         local_file:str,
                         preproc_root='l3s_viirs', 
                         field_size=(64,64), 
                         n_cores=10,
                         valid_fraction=1., 
                         write_cutouts:bool=True,
                         override_RAM=False,
                         s3_file=None, debug=False):
    """Main routine to extract and pre-process L3S data for later SST analysis
    The l3s_table is modified in place.

    Args:
        l3s_table (pandas.DataFrame): cutout table
        local_file (str): path to PreProc file
        preproc_root (str, optional): Preprocessing steps. Defaults to 'llc_std'.
        field_size (tuple, optional): Defines cutout shape. Defaults to (64,64).
        fixed_km (float, optional): Require cutout to be this size in km
        n_cores (int, optional): Number of cores for parallel processing. Defaults to 10.
        valid_fraction (float, optional): [description]. Defaults to 1..
        dlocal (bool, optional): Data files are local? Defaults to False.
        override_RAM (bool, optional): Over-ride RAM warning?
        s3_file (str, optional): s3 URL for file to write. Defaults to None.
        write_cutouts (bool, optional): 
            Write the cutouts to disk?

    Raises:
        IOError: [description]

    Returns:
        pandas.DataFrame: Modified in place table

    """
    # Preprocess options
    pdict = pp_io.load_options(preproc_root)

    # Setup for parallel
    map_fn = partial(pp_utils.preproc_image, pdict=pdict, use_mask=True)

    # Setup for dates
    uni_files = np.unique(l3s_table.ex_filename)
    if len(l3s_table) > 1000000 and not override_RAM:
        raise IOError("You are likely to exceed the RAM.  Deal")

    # Init
    pp_fields, meta, img_idx, all_sub = [], [], [], []

    # Prep LLC Table
    l3s_table = pp_utils.prep_table_for_preproc(l3s_table, 
                                                preproc_root,
                                                field_size=field_size)

    for ufile in uni_files:

        # TODO -- Rachel
        # Load up the data including the mask
        # Process the mask by our criteria
        filename = ufile
        ds = llc_io.load_llc_ds(filename, local=True)
        qmasks = np.where(np.isin(ds['quality_level'], [4,5]), 0, 1)
        sst = ds.sea_surface_temperature.values

        # Parse 
        gd_date = l3s_table.ex_filename == ufile
        sub_idx = np.where(gd_date)[0]
        all_sub += sub_idx.tolist()  # These really should be the indices of the Table
        coord_tbl = l3s_table[gd_date]

        # Add to table
        l3s_table.loc[gd_date, 'filename'] = ufile

        # Load up the cutouts
        fields = []
        masks = []
        for r, c in zip(coord_tbl.row, coord_tbl.col):
            dr = field_size[0]
            dc = field_size[1]
            #
            if (r+dr >= sst.shape[0]) or (c+dc > sst.shape[1]):
                fields.append(None)
                masks.append(None)
            else:
                fields.append(sst[r:r+dr, c:c+dc])
                masks.append(qmasks[r:r+dr, c:c+dc])
        print("Cutouts loaded for {}".format(ufile))

        # Multi-process time
        # 
        items = [item for item in zip(fields,masks,sub_idx)]

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, items,
                                             chunksize=chunksize), total=len(items)))

        # Deal with failures
        answers = [f for f in answers if f is not None]

        # Slurp
        pp_fields += [item[0] for item in answers if item is not None]
        img_idx += [item[1] for item in answers if item is not None]
        meta += [item[2] for item in answers if item is not None]

        del answers, fields, items, sst
        # Close the file
        #ds.close()

    # Fuss with indices
    ex_idx = np.array(all_sub)
    ppf_idx = []
    ppf_idx = catalog.match_ids(np.array(img_idx), ex_idx)
    
    # Write
    l3s_table = pp_utils.write_pp_fields(
        pp_fields, meta, l3s_table, ex_idx, ppf_idx,
        valid_fraction, s3_file, local_file, debug=debug,
        write_cutouts=write_cutouts)

    # Clean up
    del pp_fields

    # Upload to s3? 
    if s3_file is not None:
        ulmo_io.upload_file_to_s3(local_file, s3_file)
        print("Wrote: {}".format(s3_file))
        # Delete local?

    # Return
    return l3s_table 

