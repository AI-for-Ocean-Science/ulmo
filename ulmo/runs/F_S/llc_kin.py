""" Module to run all analysis related to fixed 144km Uniform sampling of LLC 
 144km is equivalent to 64 pixels at VIIRS sampling binned by 3
"""
import os
import numpy as np

import h5py
import pandas
from pkg_resources import resource_filename

from ulmo.llc import extract 
from ulmo.llc import uniform
from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.preproc import plotting as pp_plotting

from ulmo.ssl.train_util import option_preprocess
from ulmo.ssl import latents_extraction

from IPython import embed

tst_file = 's3://llc/Tables/test_FS_r5.0_test.parquet'
full_fileA = 's3://llc/Tables/LLC_FS_r0.5A.parquet'


def u_init_kin(tbl_file:str, debug=False, 
               resol=0.5, 
               plot=False,
               minmax_lat=None):
    """ Get the show started by sampling uniformly
    in space and and time

    Args:
        tbl_file (str): _description_
        debug (bool, optional): _description_. Defaults to True.
        resol (float, optional): _description_. Defaults to 0.5.
        plot (bool, optional): Plot the spatial distribution?
        minmax_lat (tuple, optional): Restrict on latitude
    """

    if debug:
        tbl_file = tst_file
        resol = 5.0

    # Begin 
    llc_table = uniform.coords(resol=resol, minmax_lat=minmax_lat,
                               field_size=(64,64), outfile=tbl_file)
    # Reset index                        
    llc_table.reset_index(inplace=True, drop=True)
    # Plot
    if plot:
        pp_plotting.plot_extraction(llc_table, s=1, resol=resol)

    # Temporal sampling
    if debug:
        # Extract 1 day across the full range;  ends of months
        dti = pandas.date_range('2011-09-13', periods=1, freq='2M')
    else:
        # A
        # Extract 12 days across the full range;  ends of months; every month
        dti = pandas.date_range('2011-09-13', periods=12, freq='1M')
    llc_table = extract.add_days(llc_table, dti, outfile=tbl_file)

    print(f"Wrote: {tbl_file} with {len(llc_table)} unique cutouts.")
    print("All done with init")


def u_extract_kin(tbl_file:str, debug=False, 
                  debug_local=False, 
                  root_file=None, dlocal=True, 
                  preproc_root='llc_FS'):
    """Extract 144km cutouts and resize to 64x64
    Add noise too!
    And calcualte F_S stats
    And extract divb and F_s cutouts!

    All of the above is true (JXP on 2023-01-Mar)

    Args:
        tbl_file (str): _description_
        debug (bool, optional): _description_. Defaults to False.
        debug_local (bool, optional): _description_. Defaults to False.
        root_file (_type_, optional): _description_. Defaults to None.
        dlocal (bool, optional): _description_. Defaults to False.
        preproc_root (str, optional): _description_. Defaults to 'llc_144'.
        dlocal (bool, optional): Use local files for LLC data.
    """
    FS_stat_dict = {}
    FS_stat_dict['calc_FS'] = True
    # Frontogenesis
    FS_stat_dict['Fronto_thresh'] = 2e-4 
    FS_stat_dict['Fronto_sum'] = True
    # Fronts
    FS_stat_dict['Front_thresh'] = 3e-3 

    # Giddy up (will take a bit of memory!)
    if debug:
        tbl_file = tst_file
        debug_local = True

    llc_table = ulmo_io.load_main_table(tbl_file)

    ''' # Another test
    if debug:
        # Cut down to first 2 days
        uni_date = np.unique(llc_table.datetime)
        gd_date = llc_table.datetime <= uni_date[1]
        llc_table = llc_table[gd_date]
        debug_local = True
    '''

    if debug:
        root_file = 'LLC_FS_test_preproc.h5'
    else:
        if root_file is None:
            root_file = 'LLC_FS_preproc.h5'

    # Setup
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

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
                                 calculate_kin=True,
                                 extract_kin=True,
                                 kin_stat_dict=FS_stat_dict,
                                 dlocal=dlocal,
                                 override_RAM=True)

    # Final write
    ulmo_io.write_main_table(llc_table, tbl_file)
    print("You should probably remove the PreProc/ folder")
    

def kin_nenya_eval(tbl_file:str, 
                   clobber_local=False, debug=False):
    # SSL model
    #opt_path = os.path.join(resource_filename('ulmo', 'runs'), 'SSL',
                              #'MODIS', 'v3', 'opts_96clear_ssl.json')
    opt_path = os.path.join(resource_filename('ulmo', 'runs'), 'SSL',
                              'MODIS', 'v4', 'opts_ssl_modis_v4.json')
    
    # Parse the model
    opt = option_preprocess(ulmo_io.Params(opt_path))
    model_file = os.path.join(opt.s3_outdir,
        opt.model_folder, 'last.pth')

    # Load up the table
    print(f"Grabbing table: {opt.tbl_file}")
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Grab the model
    print(f"Grabbing model: {model_file}")
    model_base = os.path.basename(model_file)
    ulmo_io.download_file_from_s3(model_base, model_file)

    # PreProc files
    pp_files = np.unique(llc_table.pp_file).tolist()

    # New Latents path
    s3_outdir = 's3://llc/Nenya/'
    latents_path = os.path.join(s3_outdir, opt.latents_folder)

    for ifile in pp_files:
        print(f"Working on {ifile}")
        data_file = os.path.basename(ifile)

        # Setup
        latents_file = data_file.replace('_preproc', '_latents')
        #if latents_file in existing_files and not clobber:
        #    print(f"Not clobbering {latents_file} in s3")
        #    continue
        s3_file = os.path.join(latents_path, latents_file) 

        # Download
        if not os.path.isfile(data_file):
            ulmo_io.download_file_from_s3(data_file, ifile)

        # Ready to write
        latents_hf = h5py.File(latents_file, 'w')

        # Read
        with h5py.File(data_file, 'r') as file:
            if 'train' in file.keys():
                train=True
            else:
                train=False

        # Train?
        if train: 
            print("Starting train evaluation")
            latents_numpy = latents_extraction.model_latents_extract(
                opt, data_file, 'train', model_base, None, None)
            latents_hf.create_dataset('train', data=latents_numpy)
            print("Extraction of Latents of train set is done.")

        # Valid
        print("Starting valid evaluation")
        latents_numpy = latents_extraction.model_latents_extract(
            opt, data_file, 'valid', model_base, None, None)
        latents_hf.create_dataset('valid', data=latents_numpy)
        print("Extraction of Latents of valid set is done.")

        # Close
        latents_hf.close()

        # Push to s3
        print("Uploading to s3..")
        ulmo_io.upload_file_to_s3(latents_file, s3_file)

        # Remove data file
        if not debug:
            os.remove(data_file)
            print(f'{data_file} removed')
    

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the LLC Table
    if flg & (2**0):
        # Debug
        #u_init_F_S('tmp', debug=True, plot=True)
        # Real deal
        u_init_kin(full_fileA, minmax_lat=(-72,57.))

    if flg & (2**1):
        #u_extract_F_S('', debug=True, dlocal=True)  # debug
        u_extract_kin(full_fileA)

    if flg & (2**2):
        kin_nenya_eval(full_fileA)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup Table
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate
    else:
        flg = sys.argv[1]

    main(flg)

# Init
# python -u llc_kin.py 1

# Extract 
# python -u llc_kin.py 2 

# SSL Evaluate
# python -u llc_kin.py 4