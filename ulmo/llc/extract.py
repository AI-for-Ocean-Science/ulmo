""" Module for uniform analyses of LLC outputs"""

import os
import numpy as np

import pandas

import xarray as xr
import h5py

from sklearn.utils import shuffle

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo.preproc import utils as pp_utils
from ulmo.preproc import extract as pp_extract
from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo.preproc import io as pp_io
from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo import defs as ulmo_defs



from IPython import embed

def add_days(llc_table:pandas.DataFrame, dti:pandas.DatetimeIndex, outfile=None):
    """Add dates to an LLC table

    Args:
        llc_table (pandas.DataFrame): [description]
        dti (pandas.DatetimeIndex): [description]
        outfile ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    
    # Check
    if 'datetime' in llc_table.keys():
        print("Dates already specified.  Not modifying")
        return llc_table

    # Do it
    llc_table['datetime'] = dti[0]
    for date in dti[1:]:
        new_tbl = llc_table[llc_table['datetime'] == dti[0]].copy()
        new_tbl['datetime'] = date
        llc_table = llc_table.append(new_tbl, ignore_index=True)

    # Write
    if outfile is not None:
        ulmo_io.write_main_table(llc_table, outfile)

    # Return
    return llc_table

def build_CC_mask(filename=None, temp_bounds=(-3, 34), 
                  field_size=(64,64)):
    """Build a CC mask for the LLC

    Args:
        filename ([type], optional): Filename for coord info. Defaults to None.
        temp_bounds (tuple, optional): Temperature bounds for masking. Defaults to (-3, 34).
        field_size (tuple, optional): Field size of cutouts. Defaults to (64,64).

    Returns:
        np.ndarray: CC_mask
    """
    # Load effectively any file
    if filename is None:
        filename = os.path.join(os.getenv('LLC_DATA'), 
                                'ThetaUVSalt',
                                'LLC4320_2011-09-13T00_00_00.nc')
        
    #sst, qual = ulmo_io.load_nc(filename, verbose=False)
    print("Using file = {} for the mask".format(filename))
    ds = xr.load_dataset(filename)
    sst = ds.Theta.values

    # Generate the masks
    mask = pp_utils.build_mask(sst, None, temp_bounds=temp_bounds)
    
    # Sum across the image
    CC_mask = pp_extract.uniform_filter(mask.astype(float), 
                                        field_size[0], 
                                        mode='constant', 
                                        cval=1.)

    # Clear
    mask_edge = np.zeros_like(mask)
    mask_edge[:field_size[0]//2,:] = True
    mask_edge[-field_size[0]//2:,:] = True
    mask_edge[:,-field_size[0]//2:] = True
    mask_edge[:,:field_size[0]//2] = True

    #clear = (CC_mask < CC_max) & np.invert(mask_edge)
    CC_mask[mask_edge] = 1.  # Masked

    # Return
    return CC_mask


def preproc_image(item, pdict):
    """
    Simple wrapper for preproc_field()

    Parameters
    ----------
    item : tuple
        field, idx
    pdict : dict
        Preprocessing dict

    Returns
    -------
    pp_field, idx, meta : np.ndarray, int, dict

    """
    # Unpack
    field, idx = item

    # Run
    pp_field, meta = pp_utils.preproc_field(field, None, **pdict)

    # Failed?
    if pp_field is None:
        return None

    # Return
    return pp_field.astype(np.float32), idx, meta


def preproc_for_analysis(llc_table:pandas.DataFrame, 
                         local_file:str,
                         preproc_root='llc_std', 
                         field_size=(64,64), 
                         n_cores=10,
                         valid_fraction=1., 
                         dlocal=False,
                         s3_file=None):
    # Preprocess options
    pdict = pp_io.load_options(preproc_root)

    # Setup for parallel
    map_fn = partial(preproc_image, pdict=pdict)

    # Setup for dates
    uni_date = np.unique(llc_table.datetime)
    if len(uni_date) > 10:
        raise IOError("You are likely to exceed the RAM.  Deal")

    # Init
    pp_fields, meta, img_idx = [], [], []

    # Prep LLC Table
    for key in ['LLC_file', 'pp_file']:
        if key not in llc_table.keys():
            llc_table[key] = ''
    llc_table['pp_root'] = preproc_root
    llc_table['field_size'] = field_size[0]
    llc_table['pp_idx'] = -1
    llc_table['pp_type'] = ulmo_defs.mtbl_dmodel['pp_type']['init']

    # Loop
    for udate in uni_date:
        # Parse filename
        filename = llc_io.grab_llc_datafile(udate, local=dlocal)

        # Allow for s3
        with ulmo_io.open(filename, 'rb') as f:
            ds = xr.open_dataset(f)
        sst = ds.Theta.values
        # Parse 
        gd_date = llc_table.datetime == udate
        sub_idx = np.where(gd_date)[0]
        coord_tbl = llc_table[gd_date]

        # Add to table
        llc_table.loc[gd_date, 'LLC_file'] = filename

        # Load up the cutouts
        fields = []
        for r, c in zip(coord_tbl.row, coord_tbl.col):
            fields.append(sst[r:r+field_size[0], c:c+field_size[1]])
        print("Cutouts loaded for {}".format(filename))

        # Multi-process time
        #sub_idx = np.arange(idx, idx+len(fields)).tolist()
        #idx += len(fields)
        # 
        items = [item for item in zip(fields,sub_idx)]

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, items,
                                             chunksize=chunksize), total=len(items)))

        # Deal with failures
        answers = [f for f in answers if f is not None]

        # Slurp
        pp_fields += [item[0] for item in answers]
        img_idx += [item[1] for item in answers]
        meta += [item[2] for item in answers]

        del answers, fields, items
        ds.close()

    # Recast
    pp_fields = np.stack(pp_fields)
    pp_fields = pp_fields[:, None, :, :]  # Shaped for training

    print("After pre-processing, there are {} images ready for analysis".format(pp_fields.shape[0]))
    
    # Reorder llc_table (probably no change)
    llc_table = llc_table.iloc[img_idx]

    # Fill up
    llc_table['pp_file'] = os.path.basename(local_file) if s3_file is None else s3_file
    # Mu
    llc_table['mean_temperature'] = [imeta['mu'] for imeta in meta]
    clms = list(llc_table.keys())
    # Others
    for key in ['Tmin', 'Tmax', 'T90', 'T10']:
        if key in meta[0].keys():
            llc_table[key] = [imeta[key] for imeta in meta]
            # Add to clms
            if key not in clms:
                clms += [key]

    # Train/validation
    n = int(valid_fraction * pp_fields.shape[0])
    idx = shuffle(np.arange(pp_fields.shape[0]))
    valid_idx, train_idx = idx[:n], idx[n:]

    # Update table
    llc_table.loc[valid_idx, 'pp_idx'] = np.arange(valid_idx.size)
    llc_table.loc[train_idx, 'pp_idx'] = np.arange(train_idx.size)
    llc_table.loc[valid_idx, 'pp_type'] = ulmo_defs.mtbl_dmodel['pp_type']['valid']
    llc_table.loc[train_idx, 'pp_type'] = ulmo_defs.mtbl_dmodel['pp_type']['train']

    # ###################
    # Write to disk (avoids holding another 20Gb in memory)
    with h5py.File(local_file, 'w') as f:
        # Validation
        f.create_dataset('valid', data=pp_fields[valid_idx].astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', data=llc_table.iloc[valid_idx].to_numpy(dtype=str).astype('S'))
        dset.attrs['columns'] = clms
        # Train
        if valid_fraction < 1:
            f.create_dataset('train', data=pp_fields[train_idx].astype(np.float32))
            dset = f.create_dataset('train_metadata', data=llc_table.iloc[train_idx].to_numpy(dtype=str).astype('S'))
            dset.attrs['columns'] = clms
    print("Wrote: {}".format(local_file))

    # Clean up
    del pp_fields

    # Upload to s3? 
    if s3_file is not None:
        ulmo_io.upload_file_to_s3(local_file, s3_file)
        print("Wrote: {}".format(s3_file))
        # Delete local?

    # Return
    return llc_table

# TODO -- Move to runs/LLC/
def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Build CC_mask for 64x64
    if flg & (2**0):  
        CC_mask = build_CC_mask(field_size=(64,64))
        # Load coords
        coord_ds = llc_io.load_coords()
        # New dataset
        ds = xr.Dataset(data_vars=dict(CC_mask=(['x','y'], CC_mask)),
                                       coords=dict(
                                           lon=(['x','y'], coord_ds.lon),
                                           lat=(['x','y'], coord_ds.lat)))
        # Write
        raise IOError("Write to s3!")
        filename = os.path.join(os.getenv('LLC_DATA'), 
                                'LLC_CC_mask_64.nc')
        ds.to_netcdf(filename)
        print("Wrote: {}".format(filename))


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # Build CC mask
    else:
        flg = sys.argv[1]

    main(flg)