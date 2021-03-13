""" Methods to read/write our PAE models """
import os

import json
import pickle

from ulmo.ood import ood
from ulmo import io as ulmo_io


def load_modis_l2(flavor='std', datadir=None, local=False):
    """Load up the MODIS L2 model (Prochaska, Cornillon & Reiman 2021)

    Args:
        flavor (str, optional): Type of model trained. Defaults to 'std'.
        datadir (str, optional): Path to data for the model. Defaults to None.
        local (bool, optional): If True, use local storage. Defaults to False.

    Returns:
        ulmo.ood.ProbablisticAutoencoder: the PAE
    """
    # Avoid circular imports

    # TODO -- Allow for s3

    #
    if flavor == 'loggrad':
        if datadir is None:
            datadir = './Models/R2019_2010_128x128_loggrad'
        filepath = 'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_loggrad.h5'
    elif flavor == 'std':
        if datadir is None:
            if local:
                datadir = os.path.join(os.getenv('SST_OOD'),
                                       'MODIS_L2', 'Models',
                                       'R2019_2010_128x128_std')
            else:  # s3
                datadir = os.path.join('s3://modis-l2', 'Models',
                                       'R2019_2010_128x128_std')
        filepath = 'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'
    
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