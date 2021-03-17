""" Script to evaluate a test of LLC data """
import os

import pandas

from ulmo.models import io as model_io
from ulmo.llc import extract 
from ulmo.llc import uniform
from ulmo import io as ulmo_io


def u_init():
    # Generate the extraction file
    resol = 0.5  # deg
    #outfile = os.path.join(os.getenv('SST_OOD'), 'LLC', 'Tables', 'test_uniform_0.5.csv')
    outfile = 's3://llc/Tables/test_uniform_r0.5_test.feather'
    if os.path.isfile(outfile):
        print("{} exists.  Am loading it.".format(outfile))
        llc_table = llc_io.load_llc_table(outfile)
    else:
        llc_table = extract.uniform_coords(resol=resol, 
                                            field_size=(64,64),
                                            outfile=outfile)
    # Plot
    extract.plot_extraction(llc_table, s=1, resol=resol)

    # Extract 6 days across the full range;  ends of months
    dti = pandas.date_range('2011-09-01', periods=6, freq='2M')
    llc_table = extract.add_days(llc_table, dti, outfile=outfile)

    print("All done")


def u_extract():
    # Giddy up (will take a bit of memory!)
    #  Move to cloud
    llc_table = ulmo_io.load_main_table(outfile)
    pp_file = os.path.join(os.getenv('SST_OOD'), 'LLC', 'PreProc', 
                            'LLC_uniform_preproc_test.h5')
    llc_table = uniform.extract_preproc_for_analysis(llc_table, outfile=pp_file)
    # Final write
    ulmo_io.write_main_table(llc_table, outfile)
    

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
        flg += 2 ** 0  # Setup coords
    else:
        flg = sys.argv[1]

    main(flg)