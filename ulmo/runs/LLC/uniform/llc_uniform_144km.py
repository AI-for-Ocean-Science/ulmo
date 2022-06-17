""" Module to run all analysis related to fixed 144km Uniform sampling of LLC 
 144km is equivalent to 64 pixels at VIIRS sampling binned by 3
"""
import os
import numpy as np

import pandas

from ulmo.llc import extract 
from ulmo.llc import uniform
from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.preproc import plotting as pp_plotting

from IPython import embed

tst_file = 's3://llc/Tables/test_uniform144_r0.5_test.parquet'
full_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
nonoise_file = 's3://llc/Tables/LLC_uniform144_r0.5_nonoise.parquet'


def u_init_144(tbl_file:str, debug=False, resol=0.5, plot=False,
               max_lat=None):
    """ Get the show started by sampling uniformly
    in space and and time

    Args:
        tbl_file (str): _description_
        debug (bool, optional): _description_. Defaults to True.
        resol (float, optional): _description_. Defaults to 0.5.
        plot (bool, optional): Plot the spatial distribution?
        max_lat (float, optional): Restrict on latitude
    """

    if debug:
        tbl_file = tst_file

    # Begin 
    llc_table = uniform.coords(resol=resol, max_lat=max_lat,
                               field_size=(64,64), outfile=tbl_file)
    # Plot
    if plot:
        pp_plotting.plot_extraction(llc_table, s=1, resol=resol)

    # Temporal sampling
    if debug:
        # Extract 6 days across the full range;  ends of months
        dti = pandas.date_range('2011-09-13', periods=6, freq='2M')
    else:
        # Extract 24 days across the full range;  ends of months; every 2 weeks
        dti = pandas.date_range('2011-09-13', periods=24, freq='2W')
    llc_table = extract.add_days(llc_table, dti, outfile=tbl_file)

    print(f"Wrote: {tbl_file} with {len(llc_table)} unique cutouts.")
    print("All done with init")


def u_extract_144(tbl_file:str, debug=False, 
                  debug_local=False, 
                  root_file=None, dlocal=True, 
                  preproc_root='llc_144'):
    """Extract 144km cutouts and resize to 64x64
    Add noise too (if desired)!

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
    llc_table = ulmo_io.load_main_table(tbl_file)

    if debug:
        # Cut down to first 2 days
        uni_date = np.unique(llc_table.datetime)
        gd_date = llc_table.datetime <= uni_date[1]
        llc_table = llc_table[gd_date]
        debug_local = True

    if debug:
        root_file = 'LLC_uniform144_test_preproc.h5'
    else:
        if root_file is None:
            root_file = 'LLC_uniform144_preproc.h5'

    # Setup
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
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
                                 #debug=debug,
                                 dlocal=dlocal,
                                 override_RAM=True)
    # Final write
    if not debug:
        ulmo_io.write_main_table(llc_table, tbl_file)
    print("You should probably remove the PreProc/ folder")
    

def u_evaluate_144(tbl_file:str, 
                   clobber_local=False, debug=False,
                   model='viirs-98'):
    
    if debug:
        tbl_file = tst_file
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Evaluate
    ulmo_evaluate.eval_from_main(llc_table,
                                 model=model)

    # Write 
    ulmo_io.write_main_table(llc_table, tbl_file)


def u_add_velocities():
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)
    
    # Velocities
    extract.velocity_stats(llc_table)

    # Write 
    ulmo_io.write_main_table(llc_table, tbl_file)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Generate the LLC Table
    if flg & (2**0):
        # Debug
        #u_init_144('tmp', debug=True, plot=True)
        # Real deal
        #u_init_144(full_file, max_lat=57.)
        u_init_144(nonoise_file, max_lat=57.)

    if flg & (2**1):
        # Debug
        #u_extract_144('', debug=True, dlocal=True)
        # Real deal
        #u_extract_144(full_file)#, debug=True)
        u_extract_144(nonoise_file, preproc_root='llc_144_nonoise',
            root_file = 'LLC_uniform144_nonoise_preproc.h5')

    if flg & (2**2):
        u_evaluate_144(full_file)

    if flg & (2**3):
        u_add_velocities()

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup Table
        flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate
        #flg += 2 ** 3  # 8 -- Velocities
    else:
        flg = sys.argv[1]

    main(flg)

# Setup
# python -u llc_uniform_144km.py 1

# Extract with noise
# python -u llc_uniform_144km.py 2 

# Evaluate -- run in Nautilus
# python -u llc_uniform_144km.py 4