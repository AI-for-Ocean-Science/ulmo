""" Module for uniform analyses of LLC outputs"""

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
        #llc_table = llc_table.append(new_tbl, ignore_index=True)
        llc_table = pandas.concat([llc_table, new_tbl], ignore_index=True)

    # Drop index
    llc_table.drop(columns=['index'], inplace=True)

    # Write
    if outfile is not None:
        assert catalog.vet_main_table(llc_table)
        ulmo_io.write_main_table(llc_table, outfile)

    # Return
    return llc_table

def build_CC_mask(filename=None, temp_bounds=(-3, 34), 
                  field_size=(64,64)):
    """Build a CC mask for the LLC

    Args:
        filename (str, optional): Filename for coord info. Defaults to None.
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


def preproc_for_analysis(llc_table:pandas.DataFrame, 
                         local_file:str,
                         preproc_root='llc_std', 
                         field_size=(64,64), 
                         fixed_km=None,
                         n_cores=10,
                         valid_fraction=1., 
                         calculate_kin=False,
                         extract_kin=False,
                         kin_stat_dict=None,
                         dlocal=False,
                         write_cutouts:bool=True,
                         override_RAM=False,
                         s3_file=None, debug=False):
    """Main routine to extract and pre-process LLC data for later SST analysis
    The llc_table is modified in place (and also returned).

    Args:
        llc_table (pandas.DataFrame): cutout table
        local_file (str): path to PreProc file
        preproc_root (str, optional): Preprocessing steps. Defaults to 'llc_std'.
        field_size (tuple, optional): Defines cutout shape. Defaults to (64,64).
        fixed_km (float, optional): Require cutout to be this size in km
        n_cores (int, optional): Number of cores for parallel processing. Defaults to 10.
        valid_fraction (float, optional): [description]. Defaults to 1..
        calculate_kin (bool, optional): Perform frontogenesis calculations?
        extract_kin (bool, optional): Extract kinematic cutouts too!
        kin_stat_dict (dict, optional): dict for guiding FS stats
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
    # Load coords?
    if fixed_km is not None:
        coords_ds = llc_io.load_coords()
        R_earth = 6371. # km
        circum = 2 * np.pi* R_earth
        km_deg = circum / 360.
    
    # Preprocess options
    pdict = pp_io.load_options(preproc_root)

    # Setup for parallel
    map_fn = partial(pp_utils.preproc_image, pdict=pdict)

    # Kinematics
    if calculate_kin:
        if kin_stat_dict is None:
            raise IOError("You must provide kin_stat_dict with calculate_kin")
        # Prep
        if 'calc_FS' in kin_stat_dict.keys() and kin_stat_dict['calc_FS']:
            map_kin = partial(kinematics.cutout_kin, 
                         kin_stats=kin_stat_dict,
                         extract_kin=extract_kin,
                         field_size=field_size[0])

    # Setup for dates
    uni_date = np.unique(llc_table.datetime)
    if len(llc_table) > 1000000 and not override_RAM:
        raise IOError("You are likely to exceed the RAM.  Deal")

    # Init
    pp_fields, meta, img_idx, all_sub = [], [], [], []
    if calculate_kin:
        kin_meta = []
    else:
        kin_meta = None
    if extract_kin:  # Cutouts of kinematic information
        Fs_fields, divb_fields = [], []

    # Prep LLC Table
    llc_table = pp_utils.prep_table_for_preproc(
        llc_table, preproc_root, field_size=field_size)
    # Loop
    #if debug:
    #    uni_date = uni_date[0:1]

    for udate in uni_date:
        # Parse filename
        filename = llc_io.grab_llc_datafile(udate, local=dlocal)

        # Allow for s3
        ds = llc_io.load_llc_ds(filename, local=dlocal)
        sst = ds.Theta.values
        # Parse 
        gd_date = llc_table.datetime == udate
        sub_idx = np.where(gd_date)[0]
        all_sub += sub_idx.tolist()  # These really should be the indices of the Table
        coord_tbl = llc_table[gd_date]

        # Add to table
        llc_table.loc[gd_date, 'filename'] = filename

        # Load up the cutouts
        fields, rs, cs, drs = [], [], [], []
        for r, c in zip(coord_tbl.row, coord_tbl.col):
            if fixed_km is None:
                dr = field_size[0]
                dc = field_size[1]
            else:
                dlat_km = (coords_ds.lat.data[r+1,c]-coords_ds.lat.data[r,c]) * km_deg
                dr = int(np.round(fixed_km / dlat_km))
                dc = dr
                # Save for kinematics
                drs.append(dr)
                rs.append(r)
                cs.append(c)
            #
            if (r+dr >= sst.shape[0]) or (c+dc > sst.shape[1]):
                fields.append(None)
            else:
                fields.append(sst[r:r+dr, c:c+dc])
        print("Cutouts loaded for {}".format(filename))

        # Multi-process time
        # 
        items = [item for item in zip(fields,sub_idx)]

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, items,
                                             chunksize=chunksize), total=len(items)))

        # Deal with failures
        answers = [f for f in answers if f is not None]
        cur_img_idx = [item[1] for item in answers]

        # Slurp
        pp_fields += [item[0] for item in answers]
        img_idx += cur_img_idx
        meta += [item[2] for item in answers]

        del answers, fields, items
        # Kinmatics
        if calculate_kin:
            # Assuming FS for now
            #if 'calc_FS' in kin_stat_dict.keys() and kin_stat_dict['calc_FS']:

            # Grab the data fields (~5 Gb RAM)
            U = ds.U.values
            V = ds.V.values
            Salt = ds.Salt.values

            # Build cutouts
            items = []
            print("Building Kinematic cutouts")
            for jj in cur_img_idx:
                # Re-index
                ii = np.where(sub_idx == jj)[0][0]
                # Saved
                r = rs[ii]
                c = cs[ii]
                dr = drs[ii]
                dc = dr
                #
                items.append(
                    (U[r:r+dr, c:c+dc],
                    V[r:r+dr, c:c+dc],
                    sst[r:r+dr, c:c+dc],
                    Salt[r:r+dr, c:c+dc],
                    jj)
                )

            #if debug:
            #    idx, FS_metrics = kinematics.cutout_F_S(items[0], FS_stats=kin_stat_dict,
            #             field_size=field_size[0])
            # Process em
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
                answers = list(tqdm(executor.map(map_kin, items,
                                             chunksize=chunksize), total=len(items)))
            kin_meta += [item[1] for item in answers]
            if extract_kin:
                Fs_fields += [item[2] for item in answers]
                divb_fields += [item[3] for item in answers]
            del answers

        ds.close()
        #embed(header='extract 223')

    # Fuss with indices
    ex_idx = np.array(all_sub)
    ppf_idx = []
    ppf_idx = catalog.match_ids(np.array(img_idx), ex_idx)

    # Write
    llc_table = pp_utils.write_pp_fields(
        pp_fields, meta, llc_table, 
        ex_idx, ppf_idx, 
        valid_fraction, s3_file, local_file,
        kin_meta=kin_meta, debug=debug, write_cutouts=write_cutouts)

    # Write kin?
    if extract_kin:
        # F_s
        Fs_local_file = local_file.replace('.h5', '_Fs.h5')
        pp_utils.write_extra_fields(Fs_fields, llc_table, Fs_local_file)
        # divb
        divb_local_file = local_file.replace('.h5', '_divb.h5')
        pp_utils.write_extra_fields(divb_fields, llc_table, divb_local_file)
    
    # Clean up
    del pp_fields

    # Upload to s3? 
    if s3_file is not None:
        ulmo_io.upload_file_to_s3(local_file, s3_file)
        print("Wrote: {}".format(s3_file))
        # Delete local?

    # Return
    return llc_table 


def velocity_stats(llc_table:pandas.DataFrame, n_cores=10): 
    """Routine to measure velocity stats for a set of cutouts

    Args:
        llc_table (pandas.DataFrame): table of cutouts
        n_cores (int, optional): Number of cores for multi-processing. Defaults to 10.
    """
    # Identify all the files to load up
    llc_files = llc_table.filename.values
    llc_files.sort()
    uni_files = np.unique(llc_files)

    # Prep
    field_size = (llc_table.field_size[0], llc_table.field_size[0])
    map_fn = partial(kinematics.cutout_vel_stat)

    # Vel keys
    vel_keys = ['U_mean', 'V_mean', 'U_rms', 'V_rms', 'UV_mean', 'UV_rms']
    for key in vel_keys:
        if key not in llc_table.keys():
            llc_table[key] = 0.

    # Loop me
    for llc_file in uni_files:
        # Allow for s3 + Lazy
        print("Loading: {}".format(llc_file))
        with ulmo_io.open(llc_file, 'rb') as f:
            ds = xr.open_dataset(f)
        # Unlazy
        U = ds.U.values
        V = ds.V.values
        # Identify all the fields with this LLC file
        cutouts = llc_table.filename == llc_file
        cutout_idx = np.where(cutouts)[0]
        coord_tbl = llc_table[cutouts]
        # Load up the cutouts
        U_fields, V_fields = [], []
        for r, c in zip(coord_tbl.row, coord_tbl.col):
            U_fields.append(U[r:r+field_size[0], c:c+field_size[1]])
            V_fields.append(V[r:r+field_size[0], c:c+field_size[1]])

        items = [item for item in zip(U_fields,V_fields,cutout_idx)]
        
        # Parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, items,
                                             chunksize=chunksize), total=len(items)))
        # Slurp
        cutout_idx = [item[0] for item in answers]
        v_stats = [item[1] for item in answers]

        # Add
        for key in vel_keys:
            llc_table.loc[cutout_idx, key] = [v_stat[key] for v_stat in v_stats]


def velocity_field(llc_table:pandas.DataFrame, vel_type:str, 
                   local_file:str, preproc_root:str, n_cores=10, 
                   valid_fraction=1., s3_file=None,
                   debug=False, dlocal=True):  
    """Routine to measure velocity stats for a set of cutouts

    Args:
        llc_table (pandas.DataFrame): table of cutouts
        local_file (str): local path to PreProc file
        vel_type (str): Velocity field type
            curl, 
        preproc_root (str): Preprocessing steps. 
        n_cores (int, optional): Number of cores for multi-processing. Defaults to 10.
        valid_fraction (float, optional): [description]. Defaults to 1..
        dlocal (bool, optional): Data files are local? Defaults to False.
        s3_file (str, optional): s3 URL for file to write. Defaults to None.
    """
    # Preprocess options
    pdict = pp_io.load_options(preproc_root)

    # Files
    uni_date = np.unique(llc_table.datetime)
    if debug:
        uni_date = uni_date[0:5]

    # Prep
    field_size = (llc_table.field_size[0], llc_table.field_size[0])
    if vel_type == 'curl':
        map_fn = partial(kinematics.cutout_curl, pdict=pdict)
    else:
        raise IOError("Not ready for vel_type={}".format(vel_type))

    pp_fields, meta, img_idx = [], [], []
    # Loop me
    for udate in uni_date:
        # Parse filename
        llc_file = llc_io.grab_llc_datafile(udate, local=dlocal)

        # Allow for s3 + Lazy
        print("Loading: {}".format(llc_file))
        ds = llc_io.load_llc_ds(llc_file, local=dlocal)

        # Unlazy
        U = ds.U.values
        V = ds.V.values
        # Identify all the cutouts with this LLC file
        cutouts = llc_table.datetime == udate
        sub_idx = np.where(cutouts)[0]
        coord_tbl = llc_table[cutouts]
        # Load up the cutouts
        U_fields, V_fields = [], []
        for r, c in zip(coord_tbl.row, coord_tbl.col):
            U_fields.append(U[r:r+field_size[0], c:c+field_size[1]])
            V_fields.append(V[r:r+field_size[0], c:c+field_size[1]])

        items = [item for item in zip(U_fields,V_fields,sub_idx)]
        
        # Parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, items,
                                             chunksize=chunksize), total=len(items)))
        # Slurp
        img_idx += [item[0] for item in answers]
        pp_fields += [item[1] for item in answers]

    # Write 
    llc_table = pp_utils.write_pp_fields(pp_fields, meta, llc_table, 
                             np.array(img_idx), 
                             np.arange(len(img_idx)),  # Dummy
                             valid_fraction,
                             s3_file, local_file, 
                             skip_meta=True)
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

    # Test vel_stats
    if flg & (2**1):  
        tbl_file = 's3://llc/Tables/test_uniform_r0.5_test.feather'
        llc_table = ulmo_io.load_main_table(tbl_file)
        velocity_stats(llc_table)




# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # Build CC mask
        flg += 2 ** 1  # Test velocity stats
    else:
        flg = sys.argv[1]

    main(flg)