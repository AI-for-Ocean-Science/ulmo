""" Module to run all analysis related to fixed 144km Uniform sampling of LLC 
 144km is equivalent to 64 pixels at VIIRS sampling binned by 3
"""
import os
import numpy as np

import pandas
import h5py


from ulmo.llc import extract 
from ulmo.llc import uniform
from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate
from ulmo.preproc import plotting as pp_plotting
from ulmo.utils import catalog as cat_utils

from ulmo.mae import analysis as enki_analysis
from ulmo.mae import cutout_analysis
from ulmo.mae import enki_utils

from IPython import embed

enki_valid_file = 's3://llc/mae/Tables/Enki_LLC_valid_nonoise.parquet'
enki_valid_noise_file = 's3://llc/mae/Tables/Enki_LLC_valid_noise.parquet'
enki_valid_noise02_file = 's3://llc/mae/Tables/Enki_LLC_valid_noise02.parquet'

def u_init_144(tbl_file:str, resol=0.5, plot=False,
               max_lat=None, rotate:float=0.25):
    """ Get the show started by sampling uniformly
    in space and and time

    Args:
        tbl_file (str): _description_
        debug (bool, optional): _description_. Defaults to True.
        resol (float, optional): 
            Typical separation of images in deg
        plot (bool, optional): Plot the spatial distribution?
        max_lat (float, optional): Restrict on latitude
        MAE (bool, optional): Generate a table for MAE?
    """
    # Begin 
    valid_table = uniform.coords(resol=resol, max_lat=max_lat,
                               field_size=(64,64), outfile=tbl_file,
                               rotate=rotate)
    # Plot
    if plot:
        pp_plotting.plot_extraction(valid_table, s=1, resol=resol)

    # Temporal sampling
    dti = pandas.date_range('2011-09-27', periods=6, freq='2M')
    valid_table = extract.add_days(valid_table, dti, outfile=tbl_file)

    print(f"Wrote: {tbl_file} with {len(valid_table)} unique cutouts.")
    print("All done with init")


def u_extract_144(tbl_file:str, root_file:str, debug=False, 
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
    

def u_evaluate_144(tbl_file:str, 
                   clobber_local=False, debug=False,
                   model='viirs-98'):
    """ Run Ulmo on the cutouts with the given model

    RAN IN JupyterHub
    """
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Evaluate
    ulmo_evaluate.eval_from_main(llc_table,
                                 model=model)

    # Write 
    assert cat_utils.vet_main_table(llc_table)
    ulmo_io.write_main_table(llc_table, tbl_file)


def inpaint(t:int, p:int, dataset:str,
            method:str, debug:bool=False, n_cores:int=10,
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
    outfile = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'Recon',
        f'Enki_{dataset}_{method}_t{t}_p{p}.h5')
    # Do it
    if not os.path.isfile(outfile) or clobber:
        cutout_analysis.inpaint_images(outfile, t, p, dataset, method=method,
                                   n_cores=n_cores, debug=debug)
    else:                            
        print(f"Found: {outfile}.  Not clobbering..")

    # RMSE time
    enki_analysis.calc_rms(t, p, dataset, method=method, debug=debug,
                           in_recon_file=outfile, clobber=rmse_clobber,
                           keys=['valid', 'inpainted', 'valid'])
                                                 

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the LLC Table
    if flg & (2**0):
        # For Enki validation
        #u_init_144(enki_valid_file, max_lat=57.)#, plot=True)
        #u_init_144(enki_valid_noise_file, max_lat=57.)#, plot=True)
        u_init_144(enki_valid_noise02_file, max_lat=57.)#, plot=True)

    if flg & (2**1):
        # Enki
        # STANDARD
        #u_extract_144(enki_valid_file,
        #              preproc_root='llc_144_nonoise', 
        #              root_file='Enki_LLC_valid_nonoise_preproc.h5') 

        # NOISE with 0.04K
        #u_extract_144(enki_valid_noise_file,
        #              preproc_root='llc_144', 
        #              root_file='Enki_LLC_valid_noise_preproc.h5') 

        # NOISE with 0.02K
        u_extract_144(enki_valid_noise02_file,
                      preproc_root='llc_144_noise02', 
                      root_file='Enki_LLC_valid_noise02_preproc.h5') 

    if flg & (2**2):
        '''
        u_evaluate_144(enki_valid_file)
        u_evaluate_144(enki_valid_noise_file)
        '''
        u_evaluate_144(enki_valid_noise02_file)

    # Calculate RMS for various reconstructions
    if flg & (2**3):
        clobber = False
        debug=False

        '''
        # No noise
        for t in [10,20,35,50,75]:
            for p in [10,20,30,40,50]:
                print(f'Working on: t={t}, p={p}')
                enki_analysis.calc_rms(t, p, dataset='LLC2_nonoise', 
                                       clobber=clobber, debug=debug)

        # Noise
        for t in [10, 35, 50, 75]:
            for p in [10,20,30,40,50]:
                print(f'Working on: t={t}, p={p}')
                enki_analysis.calc_rms(t, p, dataset='LLC2_noise', 
                                       clobber=clobber, debug=debug)
        '''
        # 0.04K Noise but noisless original
        for t in [10]:
            for p in [10]:
                print(f'Working on: t={t}, p={p}')
                enki_analysis.calc_rms(t, p, dataset='LLC2_noise', 
                    method='noiseless',
                    remove_bias=False,
                    clobber=clobber, debug=debug)

    # Inpainting galore
    if flg & (2**4):
        for t, p in zip([10,20], [10,30]):
            inpaint(t, p, 'LLC2_nonoise', 'biharmonic', debug=False)#, rmse_clobber=False, clobber=False)
            inpaint(t, p, 'LLC2_nonoise', 'grid_nearest', debug=False)#, rmse_clobber=False)
            inpaint(t, p, 'LLC2_nonoise', 'grid_linear', debug=False)#, rmse_clobber=False)
            inpaint(t, p, 'LLC2_nonoise', 'grid_cubic', debug=False)#, rmse_clobber=False)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup Table(s)
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate 
        #flg += 2 ** 3  # 8 -- RMSE
        #flg += 2 ** 4  # 16 -- Inpainting
    else:
        flg = sys.argv[1]

    main(flg)

# Setup
# python -u validation_dataset.py 1

# Extract
# python -u validation_dataset.py 2

# Extract
# python -u validation_dataset.py 4


# RMSE
# python -u validation_dataset.py 8
