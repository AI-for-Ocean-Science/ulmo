""" I/O routines for SSL analysis """
import os
from pkg_resources import resource_filename

from ulmo import io as ulmo_io
from ulmo.nenya.train_util import option_preprocess

def translate_dataset(dataset:str):
    """ Translate the dataset name to the local path"""
    if 'modis' in dataset: 
        dset = 'MODIS_L2'
    elif 'viirs' in dataset: 
        dset = 'VIIRS'
    elif 'llc' in dataset: 
        dset = 'LLC'
    else:
        raise IOError(f'Bad dataset: {dataset}')
    return dset

def get_data_path(dataset:str, local:bool=True):
    if local:
        if dataset[0:3] == 'llc':
            data_path = os.getenv('OS_OGCM')
        else:
            data_path = os.getenv('OS_SST')
    else:
        if dataset[0:3] == 'llc':
            data_path = 's3://llc'
        elif 'viirs' in dataset: 
            data_path = 's3://viirs'
        else:
            raise IOError(f"Need to add this dataset: {dataset}")

    return data_path

def load_opt(nenya_model:str):
    """ Load the SSL model options

    Args:
        nenya_model (str): name of the model
            e.g. 'LLC', 'CF', 'v4', 'v5'

    Raises:
        IOError: _description_

    Returns:
        tuple: SSL options, model file (str)
    """
    # Prep
    ssl_model_file = None
    if nenya_model == 'LLC' or nenya_model == 'LLC_local':
        ssl_model_file = 's3://llc/SSL/LLC_MODIS_2012_model/SimCLR_LLC_MODIS_2012_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/last.pth'
        opt_path = os.path.join(resource_filename('ulmo', 'runs'),
                                'Nenya', 'LLC', 'experiments', 
                                'llc_modis_2012', 'opts.json')
    elif nenya_model == 'CF': 
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'Nenya', 'MODIS', 'v2', 'experiments',
            'modis_model_v2', 'opts_cloud_free.json')
    elif nenya_model == 'v4':  
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'Nenya', 'MODIS', 'v4', 'opts_ssl_modis_v4.json')
    elif nenya_model == 'v5': 
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'Nenya', 'MODIS', 'v5', 'opts_nenya_modis_v5.json')
        ssl_model_file = os.path.join(os.getenv('OS_SST'),
                                  'MODIS_L2', 'Nenya', 'models', 
                                  'v4_last.pth')  # Only the UMAP was retrained (for now)
    else:
        raise IOError("Bad model!!")

    opt = option_preprocess(ulmo_io.Params(opt_path))

    if ssl_model_file is None:
        ssl_model_file = os.path.join(opt.s3_outdir, 
                                  opt.model_folder, 'last.pth')

    # Return
    return opt, ssl_model_file
    

def latent_path(dataset:str, local:bool=True, 
                model:str='MODIS_R2019_v4/SimCLR_resnet50_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_5_cosine_warm'):
    data_path = get_data_path(dataset, local=local)

    if dataset == 'modis_redo':
        model = model.replace('v4', 'v4_REDO')

    dset = translate_dataset(dataset)

    return os.path.join(data_path, dset, 'Nenya', 'latents', model)

def table_path(dataset:str, local:bool=True):
    data_path = get_data_path(dataset, local=local)

    if local:
        dset = translate_dataset(dataset)
        return os.path.join(data_path, dset, 'Nenya', 'Tables')
    else:
        return os.path.join(data_path, 'Nenya', 'Tables')

def umap_path(dataset:str, local:bool=True): 
    data_path = get_data_path(dataset, local=local)
    dset = translate_dataset(dataset)

    return os.path.join(data_path, dset, 'Nenya', 'UMAP')