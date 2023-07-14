""" I/O routines for SSL analysis """
import os
from pkg_resources import resource_filename

from ulmo import io as ulmo_io
from ulmo.nenya.train_util import option_preprocess

def load_opt(model:str):
    """ Load the SSL model options

    Args:
        model (str): name of the model

    Raises:
        IOError: _description_

    Returns:
        tuple: SSL options, model file (str)
    """
    # Prep
    model_file = None
    if model == 'LLC' or model == 'LLC_local':
        model_file = 's3://llc/SSL/LLC_MODIS_2012_model/SimCLR_LLC_MODIS_2012_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/last.pth'
        opt_path = os.path.join(resource_filename('ulmo', 'runs'),
                                'Nenya', 'LLC', 'experiments', 
                                'llc_modis_2012', 'opts.json')
    elif model == 'CF': 
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'Nenya', 'MODIS', 'v2', 'experiments',
            'modis_model_v2', 'opts_cloud_free.json')
    elif model == 'v4':  
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'Nenya', 'MODIS', 'v4', 'opts_ssl_modis_v4.json')
    elif model == 'v5': 
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'Nenya', 'MODIS', 'v5', 'opts_nenya_modis_v5.json')
        model_file = os.path.join(os.getenv('OS_SST'),
                                  'MODIS_L2', 'Nenya', 'models', 
                                  'v5_last.pth')
    else:
        raise IOError("Bad model!!")

    opt = option_preprocess(ulmo_io.Params(opt_path))

    if model_file is None:
        model_file = os.path.join(opt.s3_outdir, 
                                  opt.model_folder, 'last.pth')

    # Return
    return opt, model_file
    