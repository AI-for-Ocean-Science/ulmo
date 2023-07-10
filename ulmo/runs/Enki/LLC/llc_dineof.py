""" Module for methods related to running DINEOF on LLC data
"""
import os
import numpy as np

import pandas
import h5py

import xarray

from astropy import units
from astropy.coordinates import SkyCoord, match_coordinates_sky

from ulmo.llc import extract 
from ulmo.llc import io as llc_io
from ulmo.llc import slurp
from ulmo import io as ulmo_io
from ulmo.preproc import plotting as pp_plotting
from ulmo.utils import catalog as cat_utils
from ulmo.scripts import grab_llc
from ulmo.scripts import enki_reconstruct
from ulmo.mae import patch_analysis

from IPython import embed

enki_dineof_file = 's3://llc/mae/Tables/Enki_LLC_DINOEF.parquet'

def dineof_init(tbl_file:str, debug=False, plot=False,
                field_size=(64,64), resol=144, max_lat=21):
    """ Get the show started by sampling only at 118E, 21N

    Args:
        tbl_file (str): _description_
        debug (bool, optional): _description_. Defaults to True.
        plot (bool, optional): Plot the spatial distribution?
    """
    # Load up CC_mask
    CC_mask = llc_io.load_CC_mask(field_size=field_size, local=True)

    # Cut
    CC_max=1e-4
    good_CC = CC_mask.CC_mask.values < CC_max
    good_CC_idx = np.where(good_CC)

    # Build coords
    llc_lon = CC_mask.lon.values[good_CC].flatten()
    llc_lat = CC_mask.lat.values[good_CC].flatten()
    print("Building LLC SkyCoord")
    llc_coord = SkyCoord(llc_lon*units.deg + 180.*units.deg, 
                         llc_lat*units.deg, 
                         frame='galactic')

    # Cross-match
    print("Cross-match")
    china_sea = SkyCoord((180+118)*units.deg, 21*units.deg, frame='galactic')
    idx, sep2d, _ = match_coordinates_sky(china_sea, llc_coord, nthneighbor=1)


    # Build the table
    dineof_table = pandas.DataFrame()
    dineof_table['lat'] = [llc_lat[idx]]  # Center of cutout
    dineof_table['lon'] = llc_lon[idx]  # Center of cutout

    dineof_table['row'] = good_CC_idx[0][idx] - field_size[0]//2 # Lower left corner
    dineof_table['col'] = good_CC_idx[1][idx] - field_size[0]//2 # Lower left corner


    # Plot
    if plot:
        pp_plotting.plot_extraction(dineof_table, s=1, resol=resol)

    # Temporal sampling
    dti = pandas.date_range('2011-09-27', periods=180, freq='1D')
    dineof_table = extract.add_days(dineof_table, dti, outfile=tbl_file)

    print(f"Wrote: {tbl_file} with {len(dineof_table)} unique cutouts.")
    print("All done with init")


def dineof_extract(tbl_file:str, root_file:str, debug=False, 
                  debug_local=False, 
                  dlocal=True, 
                  preproc_root='llc_144'):
    """Extract 144km cutouts and resize to 64x64
    Add noise too (if desired)!

    Args:
        tbl_file (str): _description_
        root_file (_type_, optional): 
            Output file. 
        debug (bool, optional): _description_. Defaults to False.
        debug_local (bool, optional): _description_. Defaults to False.
        preproc_root (str, optional): _description_. Defaults to 'llc_144'.
        dlocal (bool, optional): Use local files for LLC data.
    """

    # Giddy up (will take a bit of memory!)
    llc_table = ulmo_io.load_main_table(tbl_file)

    if debug:
        # Cut down to first 2 days
        uni_date = np.unique(llc_table.datetime)
        gd_date = llc_table.datetime <= uni_date[1]
        llc_table = llc_table[gd_date]
        debug_local = True

    if debug:
        root_file = 'Enki_LLC_uniform144_test_preproc.h5'

    # Setup
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    pp_s3_file = pp_s3_file.replace('PreProc', 'mae/PreProc')
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    print(f"Outputting to: {pp_s3_file}")

    # Run it
    if debug_local:
        pp_s3_file = None  
    # Check indices
    assert np.all(np.arange(len(llc_table)) == llc_table.index)
    # Do it
    extract.preproc_for_analysis(llc_table, 
                                 pp_local_file,
                                 fixed_km=144.,
                                 preproc_root=preproc_root,
                                 s3_file=pp_s3_file,
                                 debug=debug,
                                 dlocal=dlocal,
                                 override_RAM=True)
    # Final write
    assert cat_utils.vet_main_table(llc_table)
    if not debug:
        ulmo_io.write_main_table(llc_table, tbl_file)
    print("You should probably remove the PreProc/ folder")
    

def dineof_ncfile(tbl_file:str, root_file:str, out_file:str, debug=False,
                  add_mask:bool=True):
    """ Create a DINEOF compatible nc file

    Args:
        tbl_file (str): 
            Table file
        root_file (str): 
            Root file for the pre-processed LLC data
        out_file (str): 
            Output file for the nc file
        debug (bool, optional): _description_. Defaults to False.
        add_mask (bool, optional): _description_. Defaults to False.
    """

    # Load up
    dineof = ulmo_io.load_main_table(tbl_file)

    # Images
    if debug:
        images = np.random.rand(len(dineof),64,64)
        mask = np.random.randint(2, size=(64,64))
        add_mask = True
    else:
        f = h5py.File(root_file, 'r')
        images = f['valid'][:,0,...]
        mask = np.zeros_like(images[0,...])


    # Grab coords
    field_size = (images.shape[-1], images.shape[-1])
    CC_mask = llc_io.load_CC_mask(field_size=field_size, local=True)


    lats = CC_mask.lat.values[dineof.row[0]:dineof.row[0]+field_size[1], 
                              dineof.col[0]+field_size[0]//2]
    lons = CC_mask.lon.values[dineof.row[0]+field_size[1]//2,
        dineof.col[0]:dineof.col[0]+field_size[1]]
                                                
    # Nc file
    da = xarray.DataArray(images, coords=[dineof.datetime.values, lats, lons],
                          dims=['time', 'lat', 'lon'])
    ds_dict = {'SST': da}

    if add_mask:
        da_mask = xarray.DataArray(mask, coords=[lats, lons],
                          dims=['lat', 'lon'])
        ds_dict['mask'] = da_mask
    ds = xarray.Dataset(ds_dict)
    #ds.to_netcdf(out_file, engine='h5netcdf')
    ds.to_netcdf(out_file, engine='netcdf4')
    print(f'Wrote: {out_file}')

def llc_grab_missing():
    """ Grab the missing LLC files
    """
    pargs = grab_llc.parser(['12','--var', 'Theta,U,V,W,Salt', '--istart', '46'])
    grab_llc.main(pargs)

def dineof_prep_enki(p_sz:int=4):
    # Read nc files
    orig_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'DINEOF',
                             'Enki_LLC_orig.nc')
    ds_orig = xarray.open_dataset(orig_file)
    orig_imgs = np.asarray(ds_orig.variables['SST'])
    # Reshape
    orig_imgs = orig_imgs.reshape((orig_imgs.shape[0], 1, 
                                   orig_imgs.shape[1],
                                   orig_imgs.shape[2]))

    for p in [10, 20, 30, 40, 50]:
        # open files
        #print(f'Working on: {dineof_file}')
        #ds_recon = xarray.open_dataset(dineof_file)
        mask_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'Recon',
            f'mae_mask_t75_p{p}.h5')
        f_ma = h5py.File(mask_file, 'r')

        dineof_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'DINEOF',
            f'Enki_LLC_DINEOF_p{p}.nc')
        print(f'Working on: {dineof_file}')
        preproc_file = dineof_file.replace('DINEOF_', 'DINEOF_pproc_')
        preproc_file = preproc_file.replace('nc', 'h5')

        # Extract
        #recon_imgs = np.asarray(ds_recon.variables['sst_filled'])
        mask_imgs = []
        for i in range(180):
            mask_img = f_ma['valid'][i,0,...]
            patches = patch_analysis.find_patches(mask_img, p_sz=p_sz)
            #
            mask_patch_img = np.zeros((mask_img.shape[0]//p_sz, 
                                       mask_img.shape[1]//p_sz))
            for patch in patches:
                i, j = np.unravel_index(patch, mask_img.shape)
                mask_patch_img[i//p_sz, j//p_sz] = 1.

            mask_imgs.append(mask_patch_img)
        mask_imgs = np.asarray(mask_imgs)
        
        
        # Write as hdf5
        with h5py.File(preproc_file, 'w') as f:
            # Validation
            f.create_dataset('valid', data=orig_imgs.astype(np.float32))
            # Metadata
            #dset = f.create_dataset('valid_metadata', data=main_tbl.iloc[valid_idx].to_numpy(dtype=str).astype('S'))
            #dset.attrs['columns'] = clms

            # Masks
            f.create_dataset('masks', data=mask_imgs.astype(np.float32))

def dineof_enki_reconstruct(debug:bool=False):
    # On Nautilus
    #aws --endpoint https://s3-west.nrp-nautilus.io s3 cp s3://llc/mae/mae_pretrain_ddp_mask20/checkpoint-254.pth ./;
    #cp ulmo/mae/correct_helpers.py /opt/conda/lib/python3.10/site-packages/timm/models/layers/helpers.py;
    args = enki_reconstruct.get_args_parser()
    for p in [10, 20, 30, 40, 50]:
        largs = ['--data_path', f'Enki_LLC_DINEOF_pproc_p{p}.nc',
                '--output_dir', 'output', 
                '--resume', 'checkpoint-254.pth', 
                '--upload_path', f's3://llc/mae/DINEOF/Enki_LLC_DINEOF_enki_p{p}.nc',
                '--mask_upload_path', f's3://llc/mae/DINEOF/Enki_LLC_DINEOF_mask_p{p}.nc']
        if not debug:
            largs += ['--use_masks']
        pargs = args.parse_args(largs)
        enki_reconstruct.main(pargs)
    
    pass

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the LLC Table
    if flg & (2**0):
        dineof_init(enki_dineof_file)

    # Extract
    if flg & (2**1):
        dineof_extract(enki_dineof_file,
                      root_file='Enki_LLC_DINEOF_preproc.h5') 

    # Generate nc file
    if flg & (2**2):
        dineof_ncfile(enki_dineof_file,
                      'PreProc/Enki_LLC_DINEOF_preproc.h5',
                      'Enki_LLC_DINEOF_4.nc') # NetCDF4
                      #debug=True) 

    # Grab missing file
    if flg & (2**3):
        llc_grab_missing()

    # Reconstruct with Enki
    if flg & (2**4):
        # Only run this once!
        #dineof_prep_enki()

        # Then this
        dineof_enki_reconstruct()#debug=True)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup Table(s)
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- nc file
        #flg += 2 ** 3  # 8 -- Grab missing LLC file(s)
        #flg += 2 ** 4  # 16 -- Reconstruct with Enki
    else:
        flg = sys.argv[1]

    main(flg)

# Init
# python -u llc_dineof.py 1

# Missing LLC
# python -u llc_dineof.py 8
