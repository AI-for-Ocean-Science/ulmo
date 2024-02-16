""" Module to enable VIIRS reconstructions
"""
import os
import numpy as np

import h5py
import pandas

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo.llc import io as llc_io
from ulmo.llc import extract 
from ulmo.llc import uniform

from ulmo.preproc import plotting as pp_plotting
from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils

from ulmo.mae import analysis as enki_analysis
from ulmo.mae import enki_utils
from ulmo.mae import cutout_analysis
from ulmo.mae.cutout_analysis import rms_images
from ulmo.preproc import io as pp_io 
from ulmo.viirs import extract as viirs_extract
from ulmo.modis import utils as modis_utils


from IPython import embed

llc_tst_file = 's3://llc/Tables/test_uniform144_r0.5_test.parquet'
llc_full_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
llc_nonoise_file = 's3://llc/Tables/LLC_uniform144_r0.5_nonoise.parquet'

llc_tot1km_tbl_file = 's3://llc/Tables/LLC_1km_r0.5.parquet'

# MAE
mae_tst_nonoise_file = 's3://llc/mae/Tables/MAE_uniform144_test.parquet'
#mae_nonoise_file = 's3://llc/mae/Tables/MAE_uniform144_nonoise.parquet'
mae_valid_nonoise_tbl_file = 's3://llc/mae/Tables/MAE_LLC_valid_nonoise.parquet'
mae_valid_nonoise_file = 's3://llc/mae/PreProc/MAE_LLC_valid_nonoise_preproc.h5'
mae_img_path = 's3://llc/mae/PreProc'

ogcm_path = os.getenv('OS_OGCM')
if ogcm_path is not None:
    enki_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki')
    local_mae_valid_nonoise_file = os.path.join(enki_path, 'PreProc', 'MAE_LLC_valid_nonoise_preproc.h5')

# VIIRS
sst_path = os.getenv('OS_SST')
if sst_path is not None:
    viirs_file = os.path.join(sst_path, 'VIIRS', 'Tables', 'VIIRS_all_98clear_std.parquet')
    viirs_100_file = os.path.join(sst_path, 'VIIRS', 'Tables', 'VIIRS_all_100clear_std.parquet')
    viirs_100_s3_file = os.path.join('s3://viirs', 'Tables', 'VIIRS_all_100clear_std.parquet')
    viirs_100_img_file = os.path.join(sst_path, 'VIIRS', 'PreProc', 'VIIRS_all_100clear_preproc.h5')


def gen_llc_1km_table(tbl_file:str, debug:bool=False, 
                      resol:float=0.5, max_km:float=1.2, 
                      max_lat:float=None, plot:bool=True):
    """ Generate table for cutouts on ~1km scale 

    Args:
        tbl_file (str): Output name for Table. Should be in s3
        debug (bool, optional): _description_. Defaults to True.
        resol (float, optional): 
            Typical separation of images in deg
        max_lat (float, optional): Restrict on latitude
    """
    raise NotImplementedError("Need to update")
    # Figure out lat range
    coords_ds = llc_io.load_coords()
    R_earth = 6371. # km
    circum = 2 * np.pi* R_earth
    km_deg = circum / 360.

    gd_lat = km_pix <= max_km

    # Begin
    llc_table = uniform.coords(
        resol=resol, max_lat=max_lat, min_lat=min_lat,
        field_size=(64,64), outfile=tbl_file)

    # Plot
    if plot:
        pp_plotting.plot_extraction(
            llc_table, s=1, resol=resol)

    # Temporal sampling
    if debug:
        # Extract 6 days across the full range;  ends of months
        dti = pandas.date_range('2011-09-13', periods=6, freq='2M')
    else:
        # Extract 52 days across the full range;  every 1 week
        dti = pandas.date_range('2011-09-13', periods=52, freq='1W')
    llc_table = extract.add_days(llc_table, dti, outfile=tbl_file)

    # Measure DT only
    llc_table = extract.preproc_for_analysis(
        llc_table, preproc_root='llc_std', dlocal=True,
        debug=debug)

    # Vet
    assert cat_utils.vet_main_table(llc_table)

    # Write 
    ulmo_io.write_main_table(llc_table, tbl_file)

    print(f"Wrote: {tbl_file} with {len(llc_table)} unique cutouts.")
    print("All done with init")


def balance_cutouts_log10DT(tot_tbl_file:str, 
                       balanced_tbl_file:str,
                       ncutouts:int,
                       nbins:int=10,
                       debug=False): 
    """ Generate a log10 DT balanced 

    Args:
        tot_tbl_file (str): _description_
        balanced_tbl_file (str): _description_
        ncutouts (int): _description_
        debug (bool, optional): _description_. Defaults to False.
    """
    raise NotImplementedError("Need to update")
    # Giddy up (will take a bit of memory!)
    tot_tbl = ulmo_io.load_main_table(tot_tbl_file)

    # Steps
    log10DT = np.log10(tot_tbl.DT)
    logDT_steps = np.linspace(log10DT.min(),
                              log10DT.max(), 
                              nbins+1)
    nper_bin = ncutouts // nbins

    # Loop on bins                        
    save_idx = []
    for kk in range(nbins):
        # In bin
        in_bin = (log10DT > logDT_steps[kk]) & (
            log10DT <= logDT_steps[kk+1])
        idx = np.where(in_bin)[0]
        # Check
        assert len(idx) > nper_bin
        # Random choice
        save_idx.append(np.random.choice(
            idx, nper_bin, replace=False))
    save_idx = np.concatenate(save_idx)

    # 
    new_tbl = tot_tbl.iloc[save_idx]
    new_tbl.reset_index(inplace=True)

    # Vet
    assert cat_utils.vet_main_table(new_tbl)

    # Write 
    ulmo_io.write_main_table(new_tbl, balanced_tbl_file)



def gen_viirs_images(debug:bool=False):
    """ Generate a file of cloud free VIIRS images
    """
    # Load
    viirs = ulmo_io.load_main_table(viirs_file)

    # Cut on CC
    all_clear = np.isclose(viirs.clear_fraction, 0.)
    viirs_100 = viirs[all_clear].copy()


    # Generate images
    uni_pps = np.unique(viirs_100.pp_file)
    if debug:
        uni_pps=uni_pps[0:2]

    the_images = []
    for pp_file in uni_pps:
        print(f'Working on {os.path.basename(pp_file)}')
        # Go local
        local_file = os.path.join(sst_path, 'VIIRS', 'PreProc', 
                                  os.path.basename(pp_file))
        f = h5py.File(local_file, 'r')
        if 'train' in f.keys():
            raise IOError("train should not be found")
        # Load em all (faster)
        data = f['valid'][:]
        in_file = viirs_100.pp_file == pp_file
        idx = viirs_100.pp_idx[in_file].values

        data = data[idx,...]
        the_images.append(data)

    # Write
    the_images = np.concatenate(the_images)
    with h5py.File(viirs_100_img_file, 'w') as f:
        # Validation
        f.create_dataset('valid', data=the_images.astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', 
                                data=viirs_100.to_numpy(dtype=str).astype('S'))
        #dset.attrs['columns'] = clms
        '''
        # Train
        if valid_fraction < 1:
            f.create_dataset('train', data=pp_fields[train_idx].astype(np.float32))
            dset = f.create_dataset('train_metadata', data=main_tbl.iloc[train_idx].to_numpy(dtype=str).astype('S'))
            dset.attrs['columns'] = clms
        '''
    print("Wrote: {}".format(viirs_100_img_file))

    # Reset pp_idx
    viirs_100.pp_idx = np.arange(len(viirs_100))

    # Write
    ulmo_io.write_main_table(viirs_100, viirs_100_s3_file)

#def rmse_inpaint(t:int, p:int, debug:bool=False,
#                 clobber:bool=False):
#    """ Measure the RMSE of inpainted images
#
#    Args:
#        t (int): training percentile
#        p (int): mask percentil
#        debug (bool, optional): _description_. Defaults to False.
#    """
##    # Table time
#    viirs_100 = ulmo_io.load_main_table(viirs_100_s3_file)
#    RMS_metric = f'RMS_inpaint_t{t}_p{p}'
#    if RMS_metric in viirs_100.keys() and not clobber:
#        print(f"Found {RMS_metric} in table. Skipping..")
#        return
#    # Load up
#    local_orig_file = viirs_100_img_file
#    #recon_file = enki_utils.img_filename(t,p, local=True, dataset='VIIRS')
#    mask_file = enki_utils.mask_filename(t,p, local=True, dataset='VIIRS')
#    inpaint_file = os.path.join(
#        sst_path, 'VIIRS', 'Enki', 
#        'Recon', f'Enki_VIIRS_inpaint_t{t}_p{p}.h5')
#
#    f_orig = h5py.File(local_orig_file, 'r')
#    #f_recon = h5py.File(recon_file, 'r')
#    f_inpaint = h5py.File(inpaint_file, 'r')
#    f_mask = h5py.File(mask_file, 'r')
#
#    rms_inpaint = rms_images(f_orig, f_inpaint, f_mask, #nimgs=nimgs,
#                             keys=['valid', 'inpainted', 'valid'])
#
#
#    # Revise pp_idx
#    viirs_100.pp_idx = np.arange(len(viirs_100))
#
#    # Fill
#    viirs_100[RMS_metric] = rms_inpaint
#
#    # Vet
#    chk, disallowed_keys = cat_utils.vet_main_table(
#        viirs_100, return_disallowed=True, cut_prefix=['MODIS_'])
#    for key in disallowed_keys:
#        assert key[0:2] in ['LL','RM', 'DT']
#
#    # Write 
#    if debug:
#        embed(header='239 of mae_recons')
#    else:
#        ulmo_io.write_main_table(viirs_100, viirs_100_s3_file)


def viirs_extract_2013(debug=False, n_cores=20, 
                       nsub_files=5000,
                       ndebug_files=0,
                       local:bool=False, 
                       opt_root:str='viirs_cc15',
                       save_fields:bool=False):
    """Extract "cloud free" images for 2013

    Args:
        debug (bool, optional): [description]. Defaults to False.
        n_cores (int, optional): Number of cores to use. Defaults to 20.
        nsub_files (int, optional): Number of sub files to process at a time. Defaults to 5000.
        ndebug_files (int, optional): [description]. Defaults to 0.
        local (bool, optional): Use local files?
            Defaults to False.
    """
    # 10 cores took 6hrs
    # 20 cores took 3hrs

    if debug:
        tbl_file = 's3://viirs/Tables/VIIRS_2013_tst.parquet'
    else:
        tbl_file = 's3://viirs/Tables/VIIRS_2013_cc15.parquet'
    # Pre-processing (and extraction) settings
    pdict = pp_io.load_options(opt_root)
    
    # 2013 
    print("Grabbing the file list")
    all_viirs_files = ulmo_io.list_of_bucket_files('viirs')
    files = []
    bucket = 's3://viirs/'
    for ifile in all_viirs_files:
        if 'data/2013' in ifile:
            files.append(bucket+ifile)

    # Output
    if debug:
        save_path = ('VIIRS_2013'
                 '_{}clear_{}x{}_tst_inpaint.h5'.format(pdict['clear_threshold'],
                                                    pdict['field_size'],
                                                    pdict['field_size']))
    else:                                                
        save_path = ('VIIRS_2013'
                 '_{}clear_{}x{}_inpaint.h5'.format(pdict['clear_threshold'],
                                                    pdict['field_size'],
                                                    pdict['field_size']))
    s3_filename = 's3://viirs/Extractions/{}'.format(save_path)


    # Setup for preproc
    map_fn = partial(viirs_extract.extract_file,
                     field_size=(pdict['field_size'], pdict['field_size']),
                     CC_max=1.-pdict['clear_threshold'] / 100.,
                     nadir_offset=pdict['nadir_offset'],
                     temp_bounds=tuple(pdict['temp_bounds']),
                     nrepeat=pdict['nrepeat'],
                     sub_grid_step=pdict['sub_grid_step'],
                     inpaint=True)

    # Local file for writing
    f_h5 = h5py.File(save_path, 'w')
    print("Opened local file: {}".format(save_path))
    
    nloop = len(files) // nsub_files + ((len(files) % nsub_files) > 0)
    metadata = None
    for kk in range(nloop):
        # Zero out
        fields, inpainted_masks = None, None
        #
        i0 = kk*nsub_files
        i1 = min((kk+1)*nsub_files, len(files))
        print('Files: {}:{} of {}'.format(i0, i1, len(files)))
        sub_files = files[i0:i1]

        # Local?
        if local:
            local_sub = []
            path = '/tank/xavier/Oceanography/data/VIIRS'
            for ifile in sub_files:
                tmp = ifile.split('/')
                new_file = os.path.join(path, '2013', tmp[5], os.path.basename(ifile))
                local_sub.append(new_file)
            sub_files = local_sub

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(sub_files) // n_cores if len(sub_files) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, sub_files,
                                             chunksize=chunksize), total=len(sub_files)))

        # Trim None's
        answers = [f for f in answers if f is not None]
        fields = np.concatenate([item[0] for item in answers])
        inpainted_masks = np.concatenate([item[1] for item in answers])
        if metadata is None:
            metadata = np.concatenate([item[2] for item in answers])
        else:
            metadata = np.concatenate([metadata]+[item[2] for item in answers], axis=0)
        del answers

        # Write
        if kk == 0:
            if save_fields:
                f_h5.create_dataset('fields', data=fields, 
                                compression="gzip", chunks=True,
                                maxshape=(None, fields.shape[1], fields.shape[2]))
            f_h5.create_dataset('inpainted_masks', data=inpainted_masks,
                                compression="gzip", chunks=True,
                                maxshape=(None, inpainted_masks.shape[1], inpainted_masks.shape[2]))
        else:
            # Resize
            for key in ['fields', 'inpainted_masks']:
                if not save_fields and key == 'fields':
                    continue
                f_h5[key].resize((f_h5[key].shape[0] + fields.shape[0]), axis=0)
            # Fill
            if save_fields:
                f_h5['fields'][-fields.shape[0]:] = fields
            f_h5['inpainted_masks'][-fields.shape[0]:] = inpainted_masks
    

    # Metadata
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 
               'clear_fraction']
    dset = f_h5.create_dataset('metadata', data=metadata.astype('S'))
    dset.attrs['columns'] = columns
    # Close
    f_h5.close() 

    # Table time
    viirs_table = pandas.DataFrame()
    viirs_table['filename'] = [item[0] for item in metadata]
    viirs_table['row'] = [int(item[1]) for item in metadata]
    viirs_table['col'] = [int(item[2]) for item in metadata]
    viirs_table['lat'] = [float(item[3]) for item in metadata]
    viirs_table['lon'] = [float(item[4]) for item in metadata]
    viirs_table['clear_fraction'] = [float(item[5]) for item in metadata]
    viirs_table['field_size'] = pdict['field_size']
    basefiles = [os.path.basename(ifile) for ifile in viirs_table.filename.values]
    viirs_table['datetime'] = modis_utils.times_from_filenames(basefiles, ioff=-1, toff=0)
    viirs_table['ex_filename'] = s3_filename

    # Vet
    assert cat_utils.vet_main_table(viirs_table)

    # Final write
    ulmo_io.write_main_table(viirs_table, tbl_file)
    
    # Push to s3
    print("Pushing to s3")
    ulmo_io.upload_file_to_s3(save_path, s3_filename)

def viirs_inpaint(t:int, p:int, dataset:str,
            debug:bool=False, n_cores:int=10,
            clobber:bool=False, rmse_clobber:bool=False):
    """ Wrapper to inpaint_images

    Args:
        t (int): training percentile
        p (int): mask percentile
        dataset (str): dataset ['VIIRS', 'LLC', 'LLC2_nonoise]
        method (str, optional): Inpainting method. Defaults to 'biharmonic'.
        debug (bool, optional): Debug?. Defaults to False.
        patch_sz (int, optional): patch size. Defaults to 4.
        n_cores (int, optional): number of cores. Defaults to 10.
        clobber (bool, optional): Clobber? Defaults to False.
        rmse_clobber (bool, optional): Clobber? Defaults to False.
    """
    # Outfile
    outfile = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Enki', 'Recon',
        f'Enki_{dataset}_inpaint_t{t}_p{p}.h5')
    # Do it
    if not os.path.isfile(outfile) or clobber:
        cutout_analysis.inpaint_images(outfile, t, p, dataset, 
                                   n_cores=n_cores, debug=debug,
                                   nsub_files=100000)
    else:                            
        print(f"Found: {outfile}.  Not clobbering..")

    # RMSE time
    enki_analysis.calc_rms(t, p, dataset, debug=debug, method='inpaint',
                           in_recon_file=outfile, clobber=rmse_clobber,
                           keys=['valid', 'inpainted', 'valid'])
                                                 


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the VIIRS images
    if flg & (2**0):
        gen_llc_1km_table(llc_tot1km_tbl_file)

    # Inpaint VIIRS images
    if flg & (2**1):
        '''
        for t in [10,20]:
            for p in [10,20,30,40]:
                if t==20 and p==10:
                    continue
                inpaint(t, p, 'VIIRS', debug=False)
        # Redo 10,10
        for t in [10]:
            for p in [10]:
                inpaint(t, p, 'VIIRS', debug=False,
                        clobber=True, rmse_clobber=True)
        '''

        # More
        for t in [20]:
            for p in [50]:
                # This does RMSE too
                viirs_inpaint(t, p, 'VIIRS', debug=False)

    # Extract with clouds at ~10%
    if flg & (2**3):
        viirs_extract_2013(local=True, save_fields=True)

    # RMSE recalc
    if flg & (2**4):

        debug = False
        dataset = 'VIIRS'

        for t in [20]:
            for p in [50]:
                print(f'Working on: t={t}, p={p}')
                #outfile = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Enki', 'Recon',
                #    f'Enki_{dataset}_inpaint_t{t}_p{p}.h5')
                #enki_analysis.calc_rms(t, p, dataset, debug=debug, 
                #           in_recon_file=outfile, clobber=True,
                #           keys=['valid', 'inpainted', 'valid'])
                enki_analysis.calc_rms(t, p, dataset, debug=debug, 
                           clobber=True)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Generate the total table
        #flg += 2 ** 1  # 2 -- Inpaint  + RMSE
        #flg += 2 ** 3  # 8 -- Extract VIIRS to CC15 (for cloud masks!)
        #flg += 2 ** 4  # 16 -- RMSE
    else:
        flg = sys.argv[1]

    main(flg)

# Inpaint
# python -u enki_viirs.py 2