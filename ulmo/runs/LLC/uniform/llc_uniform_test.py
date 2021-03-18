""" Script to evaluate a test of LLC data """
import os
import numpy as np
from urllib.parse import urlparse

import pandas

from ulmo.models import io as model_io
from ulmo.llc import extract 
from ulmo.llc import uniform
from ulmo import io as ulmo_io
from ulmo import defs as ulmo_defs

from IPython import embed

tbl_file = 's3://llc/Tables/test_uniform_r0.5_test.feather'

def u_init():
    # Generate the extraction file
    resol = 0.5  # deg
    #outfile = os.path.join(os.getenv('SST_OOD'), 'LLC', 'Tables', 'test_uniform_0.5.csv')
    llc_table = uniform.coords(resol=resol, 
                               field_size=(64,64), outfile=tbl_file)
    # Plot
    extract.plot_extraction(llc_table, s=1, resol=resol)

    # Extract 6 days across the full range;  ends of months
    dti = pandas.date_range('2011-09-01', periods=6, freq='2M')
    llc_table = extract.add_days(llc_table, dti, outfile=tbl_file)

    print("All done with init")


def u_extract(debug_local=False):

    # Giddy up (will take a bit of memory!)
    llc_table = ulmo_io.load_main_table(tbl_file)
    root_file = 'LLC_uniform_test_preproc.h5'
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    # Run it
    if debug_local:
        pp_s3_file = None  
    llc_table = extract.preproc_for_analysis(llc_table, 
                                             pp_local_file,
                                             s3_file=pp_s3_file,
                                             dlocal=False)
    # Final write
    ulmo_io.write_main_table(llc_table, tbl_file)
    

def u_evaluate():
    
    # Load table
    llc_table = ulmo_io.load_main_table(tbl_file)
    uni_pp_files = np.unique(llc_table.pp_file).tolist()
    
    # Init
    llc_table['LL'] = np.nan

    # Load model
    pae = model_io.load_modis_l2(flavor='std', local=False)
    print("Model loaded!")

    # Prep
    preproc_folder = 'PreProc'
    if not os.path.isdir(preproc_folder):
        os.mkdir(preproc_folder)
    # Output file
    output_folder = 'Evaluations'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # Loop on PreProc files
    for pp_file in uni_pp_files:
        # Parse me
        parsed_s3 = urlparse(pp_file)
        local_file = os.path.join(preproc_folder, os.path.basename(pp_file))

        # Subset
        using_pp = llc_table.pp_file == pp_file
        valid = llc_table.pp_type == ulmo_defs.mtbl_dmodel['pp_type']['valid']

        # Download preproc file for speed
        print("Downloading from s3: {}".format(pp_file))
        ulmo_io.s3.Bucket(parsed_s3.netloc).download_file(
            parsed_s3.path[1:], local_file)
        print("Done!")

        # Output file for LL (local)
        log_prob_file = os.path.join(
            output_folder, os.path.basename(local_file).replace(
                'preproc', 'log_prob'))

        # Run
        LL = pae.compute_log_probs(local_file, 'valid', 
            log_prob_file, csv=False)  
    
        # Add to table
        pp_idx = llc_table[using_pp & valid]['pp_idx']
        assert len(pp_idx) == len(LL)
        llc_table.loc[using_pp & valid, 'LL'] = LL[pp_idx]

        # Remove 
        os.remove(local_file)

    # Write table
    ulmo_io.write_main_table(llc_table, tbl_file)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

        # MMT/MMIRS
    if flg & (2**0):
        u_init()

    if flg & (2**1):
        u_extract()

    if flg & (2**2):
        u_evaluate()

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup coords
        #flg += 2 ** 1  # 2 -- Extract
        #flg += 2 ** 2  # 4 -- Evaluate
    else:
        flg = sys.argv[1]

    main(flg)