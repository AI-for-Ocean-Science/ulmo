""" Module to run all analysis related to fixed 144km Uniform sampling of LLC 
 144km is equivalent to 64 pixels at VIIRS sampling binned by 3
"""
import os
import numpy as np

import h5py
import pandas
import umap  # This needs to be here for the unpickling to work
import pickle
from pkg_resources import resource_filename

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo.llc import extract 
from ulmo.llc import uniform
from ulmo.llc import kinematics
from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo.preproc import plotting as pp_plotting
from ulmo.utils import table as table_utils
from ulmo.utils import catalog as cat_utils

from ulmo.nenya.train_util import option_preprocess
from ulmo.nenya import latents_extraction
from ulmo.nenya import nenya_umap
from ulmo.nenya import io as nenya_io

from IPython import embed

tst_file = 's3://llc/Tables/test_FS_r5.0_test.parquet'
full_fileA = 's3://llc/Tables/LLC_FS_r0.5A.parquet'
viirs98_file = 's3://viirs/Tables/VIIRS_all_98clear_std.parquet'
modis_l2_file = 's3://modis-l2/Tables/MODIS_SSL_96clear.parquet'
llc_viirs98_file = 's3://llc/Tables/llc_viirs_match.parquet'

if os.getenv('SST_OOD') is not None:
    local_viirs98_file = os.path.join(os.getenv('SST_OOD'),
                                  'VIIRS', 'Tables', 'VIIRS_all_98clear_std.parquet')

nenya_opt_path = os.path.join(resource_filename('ulmo', 'runs'), 'Nenya',
                              'MODIS', 'v4', 'opts_ssl_modis_v4.json')

FS_stat_dict = {}
FS_stat_dict['version'] = 1.0 # 2023-08-17
FS_stat_dict['calc_FS'] = True
# Frontogenesis
#FS_stat_dict['Fronto_thresh'] = 2e-4  # prior to the g factor
FS_stat_dict['Fronto_thresh'] = 5e-14 
FS_stat_dict['Fronto_sum'] = True
# Fronts
#FS_stat_dict['Front_thresh'] = 3e-3 
FS_stat_dict['Front_thresh'] = 1e-13 

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


    # Giddy up (will take a bit of memory!)
    if debug:
        tbl_file = tst_file
        debug_local = True

    llc_table = ulmo_io.load_main_table(tbl_file)

    '''
    # Another test
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

    print("WARNING: THIS WILL MUCK UP PP_IDX.  Best to avoid that!!")
    embed(header='147 FIX THE ABOVE')

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
    
def rerun_kin(tbl_file:str, F_S_datafile:str, 
              divb_datafile:str,
              dlocal:bool=True,
              debug=False, n_cores=10): 

    #if debug:
    #    tbl_file = tst_file
    #    debug_local = True

    # Load table
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Open kin files
    f_FS = h5py.File(F_S_datafile, 'r')
    f_divb = h5py.File(divb_datafile, 'r')
    
    # Check indices
    assert np.all(np.arange(len(llc_table)) == llc_table.index)
    map_kin = partial(kinematics.cutout_kin, 
                    kin_stats=FS_stat_dict,
                    extract_kin=False,
                    input_FSdivb=True)

    uni_date = np.unique(llc_table.datetime)
    all_sub = []
    for udate in uni_date:
    
        gd_date = llc_table.datetime == udate
        sub_idx = np.where(gd_date)[0]
        all_sub += sub_idx.tolist()  # These really should be the indices of the Table
        sub_tbl = llc_table[gd_date]

        # Load em up
        print("Loading up the kinematic cutouts")
        items = []
        if debug:
            nitems = min(1000, len(sub_tbl))
        else:
            nitems = len(sub_tbl)
        for idx in range(nitems):
            pidx = sub_tbl.pp_idx.values[idx]
            if pidx < 0:
                continue
            # Load
            FS_cutout = f_FS['valid'][pidx,0,...]
            divb_cutout = f_divb['valid'][pidx,0,...]
            # Append
            items.append((FS_cutout, divb_cutout, sub_idx[idx]))
        print("Done.")

        # Run it
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_kin, items,
                                             chunksize=chunksize), total=len(items)))
        kin_idx = [item[0] for item in answers]
        kin_meta = [item[1] for item in answers]

        #embed(header='231 of llc_kin')

        # Fill in
        for key in kin_meta[0].keys():
            llc_table.loc[kin_idx, key] = [imeta[key] for imeta in kin_meta]

    # Vet
    # The data types for the kinematics are a bit scrambled..
    #assert cat_utils.vet_main_table(llc_table)

    # Write
    if not debug:
        ulmo_io.write_main_table(llc_table, tbl_file)

def calc_T_SST(tbl_file:str, T_SST_datafile:str, 
              dlocal:bool=True, field_size=(64,64),
              fixed_km:float=144.,
              debug=False, n_cores=10): 

    #if debug:
    #    tbl_file = tst_file
    #    debug_local = True
    # Load up coords
    coords_ds = llc_io.load_coords()
    R_earth = 6371. # km
    circum = 2 * np.pi* R_earth
    km_deg = circum / 360.

    # Load table
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Check indices
    map_kin = partial(kinematics.cutout_kin, 
                        kin_stats=None,
                        extract_kin=True,
                        field_size=field_size[0],
                        calc_T_SST=True)

    uni_date = np.unique(llc_table.datetime)

    if debug:
        uni_date = uni_date[:2]

    all_sub = []
    T_SST_fields = []
    kin_idx = []
    for udate in uni_date:
        # Load
        filename = llc_io.grab_llc_datafile(udate, local=dlocal)
        print(f"Loading up the LLC dataset: {filename}")
        ds = llc_io.load_llc_ds(filename, local=dlocal)
        sst = ds.Theta.values
        U = ds.U.values
        V = ds.V.values 
        Salt = ds.Salt.values
    
        gd_date = llc_table.datetime == udate
        sub_idx = np.where(gd_date)[0]
        all_sub += sub_idx.tolist()  # These really should be the indices of the Table
        sub_tbl = llc_table[gd_date]

        # Load em up
        print("Loading up the kinematic cutouts")
        items = []
        if debug:
            nitems = min(1000, len(sub_tbl))
        else:
            nitems = len(sub_tbl)
        for idx in range(nitems):
            #
            pidx = sub_tbl.pp_idx.values[idx]
            if pidx < 0:
                continue
            # setup
            r = sub_tbl.row.values[idx]
            c = sub_tbl.col.values[idx]
            dlat_km = (coords_ds.lat.data[r+1,c]-coords_ds.lat.data[r,c]) * km_deg
            dr = int(np.round(fixed_km / dlat_km))
            dc = dr
            # Load items
            items.append(
                    (U[r:r+dr, c:c+dc],
                    V[r:r+dr, c:c+dc],
                    sst[r:r+dr, c:c+dc],
                    Salt[r:r+dr, c:c+dc],
                    pidx)
                )
        print("Done.")

        # Run it
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_kin, items,
                                             chunksize=chunksize), total=len(items)))
        kin_idx += [item[0] for item in answers]
        T_SST_fields += [item[2] for item in answers]

    # Repackage
    T_SST_fields = np.stack(T_SST_fields)
    T_SST_fields = T_SST_fields[:, None, :, :]  # Shaped for training
    T_SST_fields = T_SST_fields.astype(np.float32) # Recast
    
    # Deal with indexing
    new_idx = cat_utils.match_ids(np.arange(len(kin_idx)), kin_idx)

    # Write T_SST to disk
    with h5py.File(T_SST_datafile, 'w') as f:
        # Validation
        f.create_dataset(
            'valid', data=T_SST_fields[new_idx])


    # Vet
    # The data types for the kinematics are a bit scrambled..
    #assert cat_utils.vet_main_table(llc_table)

    # Write
    #if not debug:
    #    ulmo_io.write_main_table(llc_table, tbl_file)


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
    #embed(header='184 of llc_kin')
    opt = option_preprocess(ulmo_io.Params(nenya_opt_path))
    model_file = os.path.join(opt.s3_outdir,
        opt.model_folder, 'last.pth')

    # Load up the table
    print(f"Grabbing table: {tbl_file}")
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Grab the model
    model_base = os.path.basename(model_file)
    if not os.path.isfile(model_base) or clobber_local:
        print(f"Grabbing model: {model_file}")
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

def train_viirs_llc_umap(subset:str, umap_savefile:str,
                         ntrain:int=200000):

    DT_cut = None if subset == 'DTall' else subset

    # Load LLC latents
    tbl = ulmo_io.load_main_table(full_fileA)
    table = 'llc'
    llc_latents = nenya_umap.umap_subset(tbl.copy(),
                         nenya_opt_path, 
                         None, 
                         local=True,
                         DT_cut=DT_cut, 
                         DT_key = 'DT',
                         debug=False, 
                         table=table,
                         load_latents_only=True,
                         local_dataset_path=nenya_io.latent_path(table),
                         remove=False, CF=False)
    # VIIRS latents
    tbl = ulmo_io.load_main_table(local_viirs98_file)
    table = 'viirs'
    viirs_latents = nenya_umap.umap_subset(tbl.copy(),
                         nenya_opt_path, 
                         None, 
                         local=True,
                         DT_cut=DT_cut, 
                         DT_key = 'DT',
                         debug=False, 
                         table=table,
                         load_latents_only=True,
                         local_dataset_path=nenya_io.latent_path(table),
                         remove=False, CF=False)
    # Combine
    random_llc = np.random.choice(np.arange(llc_latents.shape[0]), 
                                  size=min(ntrain, llc_latents.shape[0]), 
                                  replace=False)
    random_viirs = np.random.choice(np.arange(viirs_latents.shape[0]), 
                                  size=min(ntrain, viirs_latents.shape[0]), 
                                  replace=False)
    all_latents = np.concatenate([llc_latents[random_llc,...], 
                                  viirs_latents[random_viirs,...]])
    # Train                        
    print(f"Training UMAP on a random {all_latents.shape[0]} set of the files")
    #random = np.random.choice(np.arange(nlatents), size=ntrain, 
    #                        replace=False)
    reducer_umap = umap.UMAP(random_state=42)
    latents_mapping = reducer_umap.fit(all_latents)
    print("Done..")
    pickle.dump(latents_mapping, open(umap_savefile, "wb" ) )
    print(f"Saved UMAP to {umap_savefile}")

    
def run_nenya_umap(tbl_file:str, 
               subset:str, 
               local_latents_path:str, 
               out_root:str, 
               out_table_path:str,
               table:str, 
               debug=False, 
               umap_savefile:str=None,
               local:bool=True,
               train_umap:bool=False,
               skip_vet:bool=False,
               DT_key='DT40'):
    """ Run UMAP on Nenya output

    Args:
        tbl_file (str): Full table file
        subset (str): DT subset, e.g. DT1
        local_latents_path (str): Path to the latents
        out_root (str): Root for the UMAP output table
        table (str): Descriptor of the dataset, passed to umap_subset()
        out_table_path (str): Path to the output table
            Can be s3
        umap_savefile (str, optional): UMAP save file to use. Defaults to None.
        train_umap (bool, optional): Train the UMAP? Defaults to False.
        debug (bool, optional): _description_. Defaults to False.
        local (bool, optional): _description_. Defaults to True.
        DT_key (str, optional): Key for DT. Defaults to 'DT40'.
        skip_vet (bool, optional): Skip vetting the table. Defaults to False.
            Required for LLC with <NA> kinematic values
    """

    # Load table
    tbl = ulmo_io.load_main_table(tbl_file)
    if 'DT' not in tbl.keys():
        tbl['DT'] = tbl.T90 - tbl.T10

    #embed(header='254 of llc_kin')

    # UMAP save file
    if umap_savefile is None:
        base1 = '96clear_v4'
        umap_savefile = os.path.join(
            os.getenv('SST_OOD'), 
            f'MODIS_L2/UMAP/MODIS_SSL_{base1}_{subset}_UMAP.pkl')

    # Output files
    outfile = os.path.join(
        out_table_path, f'{out_root}_{subset}.parquet')


    DT_cut = None if subset == 'DTall' else subset

    nenya_umap.umap_subset(tbl.copy(),
                         nenya_opt_path, 
                         outfile, 
                         local=local,
                         DT_cut=DT_cut, 
                         DT_key = DT_key,
                         debug=debug, 
                         table=table,
                         train_umap=train_umap, 
                         umap_savefile=umap_savefile,
                         local_dataset_path=local_latents_path,
                         skip_vet=skip_vet,
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
        #u_extract_kin('', debug=True, dlocal=True)  # debug
        u_extract_kin(full_fileA)

    if flg & (2**2):  # Nenya on correct LLC
        kin_nenya_eval(full_fileA,
                       s3_outdir='s3://llc/Nenya/')

    # Nenya on VIIRS
    if flg & (2**3):
        kin_nenya_eval(viirs98_file, 
                       s3_outdir='s3://viirs/Nenya/')

    # Nenya on alternate LLC VIIRS144
    if flg & (2**4):
        kin_nenya_eval(llc_viirs98_file,
                       s3_outdir='s3://llc/Nenya/')

    # UMAPs
    if flg & (2**5):

        '''
        # VIIRS
        run_nenya_umap(local_viirs98_file, 'DT1',
            ssl_io.latent_path('viirs'),
                   'VIIRS_Nenya', 'viirs', 
                   's3://viirs/Nenya/',
                   local=True, DT_key='DT')

        # LLC Kin
        run_nenya_umap(full_fileA, 'DT1',
            ssl_io.latent_path('llc'),
                   'LLC_A_Nenya', 'llc', 
                   's3://llc/Nenya/',
                   local=True, DT_key='DT')
        '''

        # MODIS
        run_nenya_umap(modis_l2_file, 
                   'DT1', 
                   nenya_io.latent_path('modis_redo'), 
                   'MODIS_Nenya', 
                   table_utils.path_to_tables('modis'), 
                   'modis', 
                   local=True, DT_key='DT')


    # Nenya on MODIS-L2 (test)
    if flg & (2**6):
        kin_nenya_eval(modis_l2_file, 
                       s3_outdir='s3://modis-l2/Nenya/')

    # New UMAP on VIIRS/LLC
    if flg & (2**7):

        '''
        # VIIRS on VIIRS
        subsets =  ['DT15', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5', 'DTall']
        for subset in subsets:
            run_nenya_umap(
                local_viirs98_file, subset, 
                nenya_io.latent_path('viirs'),
                'VIIRS_Nenya_98clear_v1', 
                nenya_io.table_path('viirs'), 'viirs',
                umap_savefile=os.path.join(nenya_io.umap_path('viirs'),
                    f'VIIRS_Nenya_98clear_v1_{subset}_UMAP.pkl'),
                local=True, DT_key='DT', train_umap=True)

        # VIIRS on VIIRS+LLC
        #subsets =  ['DT15', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5', 'DTall']
        subsets =  ['DT15']
        for subset in subsets:
            run_nenya_umap(
                local_viirs98_file, subset, 
                nenya_io.latent_path('viirs'),
                'VIIRS_Nenya_VIIRS_LLC', 
                nenya_io.table_path('viirs'), 'viirs',
                umap_savefile=os.path.join(nenya_io.umap_path('viirs'),
                    f'VIIRS_LLC_Nenya_v1_{subset}_UMAP.pkl'),
                local=True, DT_key='DT', train_umap=False)
        '''

        # LLC on VIIRS
        subsets =  ['DT15', 'DT0', 'DT1']#, 'DT2', 'DT4', 'DT5', 'DTall']
        for subset in subsets:
            run_nenya_umap(
                local_viirs98_file, subset, 
                nenya_io.latent_path('viirs'),
                'VIIRS_Nenya_LLC', 
                nenya_io.table_path('viirs'), 'viirs',
                umap_savefile=os.path.join(nenya_io.umap_path('llc'),
                    f'LLC_Nenya_v1_{subset}_UMAP.pkl'),
                local=True, DT_key='DT', train_umap=False)


        '''
        # LLC galore
        subsets =  ['DT15', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5', 'DTall']
        for subset in subsets:
            run_nenya_umap(
                full_fileA, subset, nenya_io.latent_path('llc'),
                'LLC_A_Nenya_v1', 
                nenya_io.table_path('llc'), 'llc',
                umap_savefile=os.path.join(nenya_io.umap_path('llc'),
                    f'LLC_Nenya_v1_{subset}_UMAP.pkl'),
                local=True, DT_key='DT', train_umap=True,
                skip_vet=True)
        '''


        '''
        # Run VIIRS UMAP on LLC
        subsets =  ['DT15', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5', 'DTall']
        for subset in subsets:
            run_nenya_umap(
                full_fileA, subset, nenya_io.latent_path('llc'),
                'LLC_A_Nenya_VIIRS', 
                nenya_io.table_path('llc', local=False), 
                'llc',
                umap_savefile=os.path.join(nenya_io.umap_path('viirs'),
                    f'VIIRS_Nenya_98clear_v1_{subset}_UMAP.pkl'),
                local=True, DT_key='DT', train_umap=False)
        '''

        '''
        # Train on LLC + VIIRS
        #subsets =  ['DT15', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5', 'DTall']
        subsets =  ['DT15']
        for subset in subsets:
            umap_savefile=os.path.join(nenya_io.umap_path('viirs'),
                    f'VIIRS_LLC_Nenya_v1_{subset}_UMAP.pkl')
            train_viirs_llc_umap(subset, umap_savefile)


        # Run VIIRS+LLC UMAP on LLC
        #subsets =  ['DT15', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5', 'DTall']
        subsets =  ['DT15']
        for subset in subsets:
            run_nenya_umap(
                full_fileA, subset, nenya_io.latent_path('llc'),
                'LLC_A_Nenya_VIIRS_LLC', 
                nenya_io.table_path('llc', local=False), 
                'llc',
                umap_savefile=os.path.join(nenya_io.umap_path('viirs'),
                    f'VIIRS_LLC_Nenya_v1_{subset}_UMAP.pkl'),
                local=True, DT_key='DT', train_umap=False,
                skip_vet=True)
        '''


    # Redo/expand kin
    if flg & (2**8):

        # Standard
        FS_file = os.path.join(os.getenv('OS_OGCM'),
                               'LLC', 'F_S', 'PreProc',
                               'LLC_FS_preproc_Fs.h5')
        divb_file = os.path.join(os.getenv('OS_OGCM'),
                               'LLC', 'F_S', 'PreProc',
                               'LLC_FS_preproc_divb.h5')
        rerun_kin(full_fileA, FS_file, divb_datafile)
                  #debug=True)


    # SST Tendency
    if flg & (2**9):
        TT_file = os.path.join(os.getenv('OS_OGCM'),
                               'LLC', 'F_S', 'PreProc',
                               'LLC_FS_preproc_T_SST.h5')
        calc_T_SST(full_fileA, TT_file) # debug=True)
                   
# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup Table
        #flg += 2 ** 1  # 2 -- Extract + Kin
        #flg += 2 ** 2  # 4 -- Evaluate LLC uniform with Nenya
        #flg += 2 ** 3  # 8 -- Evaluate VIIRS 98
        #flg += 2 ** 4  # 16 -- Evaluate LLC matched to VIIRS 98
        #flg += 2 ** 5  # 32 -- UMAP Nenya from MODIS -- This only works on 3.9!!
        #flg += 2 ** 6  # 64 -- Evaluate MODIS 96
        #flg += 2 ** 7  # 128 -- UMAPs galore
        #flg += 2 ** 8  # 256 -- Redo/expand kin 
        #flg += 2 ** 9  # 512 -- SST Tendency
    else:
        flg = sys.argv[1]

    main(flg)

# Init
# python -u llc_kin.py 1

# Extract 
# python -u llc_kin.py 2 

# SSL Evaluate
#aaaaaaaaaaaaaaaaaaaaaaaaaaaa python -u llc_kin.py 4