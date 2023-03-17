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
from ulmo.ssl import ssl_umap

from IPython import embed

tst_file = 's3://llc/Tables/test_FS_r5.0_test.parquet'
full_fileA = 's3://llc/Tables/LLC_FS_r0.5A.parquet'
viirs98_file = 's3://viirs/Tables/VIIRS_all_98clear_std.parquet'
modis_l2_file = 's3://modis-l2/Tables/MODIS_SSL_96clear.parquet'
llc_viirs98_file = 's3://llc/Tables/llc_viirs_match.parquet'

if os.getenv('SST_OOD') is not None:
    local_viirs98_file = os.path.join(os.getenv('SST_OOD'),
                                  'VIIRS', 'Tables', 'VIIRS_all_98clear_std.parquet')

nenya_opt_path = os.path.join(resource_filename('ulmo', 'runs'), 'SSL',
                              'MODIS', 'v4', 'opts_ssl_modis_v4.json')


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
    

def kin_nenya_eval(tbl_file:str, s3_outdir:str=None,
                   clobber_local=False, debug=False):
    """ Run Nenya on something

    Args:
        tbl_file (str): _description_
        s3_outdir (str, optional): 
            Path to s3 output directory.  If None, will use the
            LLC
        clobber_local (bool, optional): _description_. Defaults to False.
        debug (bool, optional): _description_. Defaults to False.
    """
    # SSL model
    #opt_path = os.path.join(resource_filename('ulmo', 'runs'), 'SSL',
                              #'MODIS', 'v3', 'opts_96clear_ssl.json')
    
    # Parse the model
    opt = option_preprocess(ulmo_io.Params(nenya_opt_path))
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
    if s3_outdir is None:
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
    
def nenya_umap(tbl_file:str, subset:str, out_path:str, out_root:str,
            table:str, s3_outdir:str, 
               clobber_local=False, debug=False, local:bool=True,
               DT_key='DT40'):

    # Load table
    tbl = ulmo_io.load_main_table(tbl_file)
    if 'DT' not in tbl.keys():
        tbl['DT'] = tbl.T90 - tbl.T10

    #embed(header='254 of llc_kin')

    # UMAP save file
    base1 = '96clear_v4'
    umap_savefile = os.path.join(
            os.getenv('SST_OOD'), 
            f'MODIS_L2/UMAP/MODIS_SSL_{base1}_{subset}_UMAP.pkl')

    # Output files
    outfile = os.path.join(
        out_path, 'Tables',
        f'{out_root}_{subset}.parquet')


    DT_cut = None if subset == 'DTall' else subset

    ssl_umap.umap_subset(tbl.copy(),
                         nenya_opt_path, 
                         outfile, 
                         local=local,
                         DT_cut=DT_cut, 
                         DT_key = DT_key,
                         debug=debug, 
                         table=table,
                         train_umap=False, 
                         umap_savefile=umap_savefile,
                         s3_outdir=s3_outdir,
                         local_dataset_path=out_path,
                         remove=False, CF=False)


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

    # Nenya on VIIRS
    if flg & (2**3):
        kin_nenya_eval(viirs98_file, 
                       s3_outdir='s3://viirs/Nenya/')

    # Nenya on LLC
    if flg & (2**4):
        kin_nenya_eval(llc_viirs98_file,
                       s3_outdir='s3://llc/Nenya/')

    if flg & (2**5):

        '''
        # VIIRS
        nenya_umap(local_viirs98_file, 'DT1',
                   os.path.join(os.getenv('SST_OOD'), 'VIIRS'),
                   'VIIRS_Nenya', 'viirs', 
                   's3://viirs/Nenya/',
                   local=True, DT_key='DT')
        '''

        # LLC Kin
        nenya_umap(full_fileA, 'DT1',
                   os.path.join(os.getenv('SST_OOD'), 'LLC'),
                   'LLC_A_Nenya', 'llc', 
                   's3://llc/Nenya/',
                   local=True, DT_key='DT')

    # Nenya on MODIS-L2 (test)
    if flg & (2**6):
        kin_nenya_eval(modis_l2_file, 
                       s3_outdir='s3://modis-l2/Nenya/')


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup Table
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate
        #flg += 2 ** 3  # 8 -- Evaluate VIIRS 98
        #flg += 2 ** 4  # 16 -- Evaluate LLC matched to VIIRS 98
        #flg += 2 ** 5  # 32 -- UMAP Nenya -- This only works on 3.9!!
        #flg += 2 ** 3  # 8 -- Evaluate MODIS 96
    else:
        flg = sys.argv[1]

    main(flg)

# Init
# python -u llc_kin.py 1

# Extract 
# python -u llc_kin.py 2 

# SSL Evaluate
# python -u llc_kin.py 4