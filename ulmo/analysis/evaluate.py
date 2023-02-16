""" Methods for evaluating Ulmo models """

import os
import numpy as np

from urllib.parse import urlparse

import pandas

from ulmo import io as ulmo_io
from ulmo.models import io as model_io
from ulmo import defs as ulmo_defs

from IPython import embed


def eval_from_main(main_table: pandas.DataFrame,
                   model='modis-l2-std',
                   clobber_local=False,
                   debug=False):
    """Evaluate a set of cutouts guided by the input table

    Args:
        main_table (pandas.DataFrame): main table describing the cutous
        model (str, optional): Name of Ulmo model to apply. Defaults to 'modis-l2-std'.
        clobber_local (bool, optional): Over-write local pre-process file, if True. Defaults to False.
        debug (bool, optional): Debug?

    Raises:
        IOError: [description]
    
    Returns:
        pandas.DataFrame: Updated main_table
    """

    # PP files
    uni_pp_files = np.unique(main_table.pp_file).tolist()

    # Init
    if 'LL' not in main_table.keys():
        main_table['LL'] = np.nan

    # Load model
    pae = model_io.load_ulmo_model(model, local=False)
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
    for kk, pp_file in enumerate(uni_pp_files):
        if debug and kk > 0:
            continue
        # Parse me
        parsed_s3 = urlparse(pp_file)
        local_file = os.path.join(preproc_folder, os.path.basename(pp_file))

        # Subset
        using_pp = main_table.pp_file == pp_file
        valid = main_table.pp_type == ulmo_defs.mtbl_dmodel['pp_type']['valid']

        # Download preproc file for speed
        if not os.path.isfile(local_file) or clobber_local:
            print("Downloading from s3: {}".format(pp_file))
            ulmo_io.s3.Bucket(parsed_s3.netloc).download_file(
                parsed_s3.path[1:], local_file)
            print("Done!")
        else:
            print("Using existing local file: {}".format(local_file))

        # Output file for LL (local)
        log_prob_file = os.path.join(
            output_folder, os.path.basename(local_file).replace(
                'preproc', 'log_prob'))

        # Run
        LL = pae.eval_data_file(local_file, 'valid',
                                log_prob_file, csv=False)

        # Add to table
        pp_idx = main_table[using_pp & valid]['pp_idx']
        try:
            assert len(pp_idx) == len(LL)
        except:
            embed(header='Problem with pp_idx in evaluate.py') 
        main_table.loc[using_pp & valid, 'LL'] = LL[pp_idx]

        # Remove
        if not debug:
            os.remove(local_file)
        print("Removed: {}".format(local_file))

    # Return
    return main_table