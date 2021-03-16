""" Script to evaluate a test of LLC data """
import os

from ulmo.models import io as model_io
from ulmo import io as ulmo_io

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