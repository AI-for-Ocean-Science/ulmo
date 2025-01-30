""" Methods to read/write our PAE models """
import os

import json
import pickle

from torch import embedding

from ulmo.ood import ood
from ulmo import io as ulmo_io

from IPython import embed


def load_ulmo_model(model_name:str, datadir=None, local=False):
    """Load up an Ulmo model including the 
        MODIS L2 model (Prochaska, Cornillon & Reiman 2021)

    Args:
        model_name (str, optional): Ulmo model trained
        datadir (str, optional): Path to data for the model. Defaults to None.
        local (bool, optional): If True, use local storage. Defaults to False.

    Returns:
        ulmo.ood.ProbablisticAutoencoder: the PAE
    """
    #
    if model_name == 'model-l2-loggrad':
        if datadir is None:
            datadir = './Models/R2019_2010_128x128_loggrad'
        filepath = 'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_loggrad.h5'
    elif model_name == 'model-l2-std':
        if datadir is None:
            if local:
                datadir = os.path.join(os.getenv('SST_OOD'),
                                       'MODIS_L2', 'Models',
                                       'R2019_2010_128x128_std')
            else:  # s3
                datadir = os.path.join('s3://modis-l2', 'Models',
                                       'R2019_2010_128x128_std')
        filepath = 'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'
    elif model_name == 'ssh-std':
        if datadir is None:
            datadir = os.path.join('s3://ssh', 'Models', 'SSH_std')
        filepath = 'PreProc/SSH_100clear_32x32_train.h5'
    elif model_name == 'viirs-test':
        if datadir is None:
            datadir = os.path.join('s3://viirs', 'Models', 'VIIRS_test')
        filepath = 'PreProc/VIIRS_2013_95clear_192x192_preproc_viirs_std_train.h5'
    elif model_name == 'viirs-98':
        if datadir is None:
            if local:
                datadir = os.path.join(os.getenv('OS_SST'),
                                       'VIIRS', 'Ulmo', 'Models',
                                       'VIIRS_std_98')
            else:
                datadir = os.path.join('s3://viirs', 'Models', 'VIIRS_std_98')
        filepath = 'PreProc/VIIRS_2013_98clear_192x192_preproc_viirs_std_train.h5'
    else:
        raise IOError("Bad Ulmo model name!!")

    # Load JSON
    json_file = os.path.join(datadir , 'model.json') 
    with ulmo_io.open(json_file, 'rt') as fh:
        model_dict = json.load(fh)
    
    # Instantiate
    pae = ood.ProbabilisticAutoencoder.from_dict(
        model_dict,
        datadir=datadir,
        filepath=filepath,
        logdir=datadir)

    # Giddy up                                            
    pae.load_autoencoder()
    pae.load_flow()

    # Load scaler
    scaler_path = os.path.join(pae.logdir, pae.stem + '_scaler.pkl')
    with ulmo_io.open(scaler_path, 'rb') as f:
        pae.scaler = pickle.load(f)
    print("scaler loaded from: {}".format(scaler_path))

    # Return
    return pae
