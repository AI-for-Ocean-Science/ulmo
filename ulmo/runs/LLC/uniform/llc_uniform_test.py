""" Script to evaluate a test of LLC data """
import os
import numpy as np

import pandas

from ulmo.models import io as model_io
from ulmo.llc import extract 
from ulmo.llc import uniform
from ulmo import io as ulmo_io

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


def u_extract():

    # Giddy up (will take a bit of memory!)
    llc_table = ulmo_io.load_main_table(tbl_file)
    root_file = 'LLC_uniform_preproc_test.h5'
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    # Run it
    llc_table = extract.preproc_for_analysis(llc_table, 
                                             pp_local_file,
                                             s3_file=pp_s3_file,
                                             dlocal=False)
    # Final write
    ulmo_io.write_main_table(llc_table, tbl_file)
    

def u_evaluate():
    # Load model
    pae = model_io.load_modis_l2(flavor='std', local=False)
    print("Model loaded!")

    # Download preproc file for speed
    preproc_folder = 'PreProc'
    if not os.path.isdir(preproc_folder):
        os.mkdir(preproc_folder)
    data_file = os.path.join(preproc_folder, 'LLC_uniform_preproc_test.h5') 
    ulmo_io.s3.Bucket('llc').download_file(data_file, data_file)

    # Output file
    output_folder = 'Evaluations'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    log_prob_file = os.path.join(output_folder, 
                                'LLC_uniform_test_std_log_prob.h5')

    # Run
    pae.compute_log_probs(data_file, 'valid', 
        log_prob_file, csv=False)  # Tends to crash on kuber

    # Remove 
    os.remove(data_file)

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

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Setup coords
        #flg += 2 ** 1  # 2 -- Extract
    else:
        flg = sys.argv[1]

    main(flg)