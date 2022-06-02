""" Module to run all analysis related to fixed 128km Uniform sampling of LLC """
import os
import numpy as np

import pandas

from ulmo.llc import extract 
from ulmo.llc import uniform
from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo.preproc import plotting as pp_plotting

from IPython import embed

tst_file = 's3://llc/Tables/test_uniform128_r0.5_test.parquet'
full_file = 's3://llc/Tables/test_uniform128_r0.5.parquet'


def u_init_128(tbl_file:str, debug=False, resol=0.5, plot=False,
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
        dti = pandas.date_range('2011-09-01', periods=6, freq='2M')
    else:
        # Extract 24 days across the full range;  ends of months; every 2 weeks
        dti = pandas.date_range('2011-09-01', periods=24, freq='2W')
    llc_table = extract.add_days(llc_table, dti, outfile=tbl_file)

    print(f"Wrote: {tbl_file} with {len(llc_table)} unique cutouts.")
    print("All done with init")


def u_extract_128(tbl_file:str, debug=False, debug_local=False,
              root_file=None, dlocal=False, preproc_root='llc_128'):

    if debug:
        tbl_file = tst_file
        debug_local = True

    # Giddy up (will take a bit of memory!)
    llc_table = ulmo_io.load_main_table(tbl_file)

    if debug:
        # Cut down to first day
        uni_date = np.unique(llc_table.datetime)
        gd_date = llc_table.datetime == uni_date[0]
        llc_table = llc_table[gd_date]

    if debug:
        root_file = 'LLC_uniform128_test_preproc.h5'
    else:
        if root_file is None:
            root_file = 'LLC_uniform128_preproc.h5'

    # Setup
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    # Run it
    if debug_local:
        pp_s3_file = None  
    extract.preproc_for_analysis(llc_table, 
                                 pp_local_file,
                                 fixed_km=128.,
                                 preproc_root=preproc_root,
                                 s3_file=pp_s3_file,
                                 debug=debug,
                                 dlocal=dlocal)
    # Final write
    ulmo_io.write_main_table(llc_table, tbl_file)
    print("You should probably remove the PreProc/ folder")
    

def u_evaluate_128(clobber_local=False, debug=False):
    
    if debug:
        tbl_file = tst_file
    # Load
    llc_table = ulmo_io.load_main_table(tbl_file)

    # Evaluate
    ulmo_evaluate.eval_from_main(llc_table)

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
        #u_init_128('tmp', debug=True, plot=True)
        # Real deal
        u_init_128(full_file, max_lat=57.)

    if flg & (2**1):
        u_extract_128('', debug=True, dlocal=True)
        #u_extract_128(full_file)

    if flg & (2**2):
        u_evaluate()

    if flg & (2**3):
        u_add_velocities()

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- Setup Table
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate
        #flg += 2 ** 3  # 8 -- Velocities
    else:
        flg = sys.argv[1]

    main(flg)