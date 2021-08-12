"""
Simple script to run Evals on all data sets in LLC and VIIRS
"""

import os
import numpy as np

from ulmo.models import io as model_io
from ulmo import io as ulmo_io
from ulmo.models.io import load_modis_l2_extension 


def run_evals_extension(model_dir, train_dir, bucket_name, clobber=False, local=False):
    """Main method to evaluate the model

    Outputs are written to hard-drive in a sub-folder
    named Evaluations/

    Args:
        model_dir (str): path of trained model.
        train_dir (str): path of training data.
        bucket_name (str): s3 bucket for evaluation.
        clobber (bool, optional): Clobber existing outputs. Defaults to False.
        local (bool, optional): Load model and data locally. 
            Otherwise use s3 storage. Defaults to False.

    Raises:
        IOError: [description]
    """

    # Load model
    pae = load_modis_l2_extension(datadir=model_dir, filepath=train_dir)
    print("Model loaded!")

    # Allow for various datasets
    
    preproc_folder = '/PreProc'
    data_list = ulmo_io.list_of_bucket_files(bucket_name, preproc_folder)
    # Generate evaluation data list
    
    # Prep
    for data_file in data_list:
        if not local:
            if not os.path.isdir('PreProc'):
                os.mkdir('PreProc')
            ulmo_io.s3.Bucket(bucket_name).download_file(data_file, data_file)
            print("Dowloaded: {} from s3".format(data_file))
        # Check
        if local:
            if not os.path.isfile(data_file):
                raise IOError("This data file does not exist! {}".format(data_file))

        # Output
        data_title = os.path.splitext(os.path.split(data_file)[1])[0]
         
        log_prob_file = f'Evaluations/{data_title}_log_probs.h5'
        if os.path.isfile(log_prob_file) and not clobber:
            print("Eval file {} exists! Skipping..".format(log_prob_file))
            continue
        
        # Run
        pae.eval_data_file(data_file, 'valid', 
                           log_prob_file, csv=False)  # Tends to crash on kuber

        # Remove local
        if not local:
            os.remove(data_file)

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Evalute the model to get log_probs.')
    parser.add_argument("--model_dir", type=str, help="Trained Model path")
    parser.add_argument("--data_dir", type=str, help="Training data path")
    parser.add_argument("--bucket_name", type=str, help="llc or viirs")
    parser.add_argument("--clobber", default=False, action="store_true", help="Debug?")
    parser.add_argument("--local", default=False, action="store_true", help="Use local storage")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs

def main():
    """ Run
    """
    import warnings
    pargs = parser()
    run_evals_extension(pargs.model_dir, pargs.data_dir, pargs.bucket_name, clobber=pargs.clobber, local=pargs.local)
