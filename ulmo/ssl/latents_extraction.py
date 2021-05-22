from __future__ import print_function

import time
import os
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import trange

import torch

from ulmo.ssl.my_util import Params, option_preprocess
from ulmo.ssl.my_util import modis_loader, set_model
from ulmo.ssl.my_util import train_modis

from IPython import embed

def model_latents_extract(opt, modis_data, model_path, save_path, save_key,
                          remove_module=True):
    """
    This function is used to obtain the latents of the training data.
    Args:
        opt: (Parameters) parameters used to create the model
        modis_data: (numpy.array) 
        model_path: (string) 
        save_path: (string)
        save_key: (string)
    """
    using_gpu = torch.cuda.is_available()
    model, _ = set_model(opt, cuda_use=using_gpu)
    if not using_gpu:
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model_dict = torch.load(model_path)

    if remove_module:
        new_dict = {}
        for key in model_dict['model'].keys():
            new_dict[key.replace('module.','')] = model_dict['model'][key]
        model.load_state_dict(new_dict)
    else:
        model.load_state_dict(model_dict['model'])
    print("Model loaded")
    
    modis_data = np.repeat(modis_data, 3, axis=1)
    num_samples = modis_data.shape[0]
    #batch_size = opt.batch_size
    batch_size = 1
    num_steps = num_samples // batch_size
    #num_steps = 1
    remainder = num_samples % batch_size
    latents_df = pd.DataFrame()
    print("Beginning to evaluate")
    with torch.no_grad():
        for i in trange(num_steps):
            image_batch = modis_data[i*batch_size: (i+1)*batch_size]
            image_tensor = torch.tensor(image_batch)
            if using_gpu:
                latents_tensor = model(image_tensor.cuda())
                latents_numpy = latents_tensor.to_cpu().numpy()
            else:
                latents_tensor = model(image_tensor)
                latents_numpy = latents_tensor.numpy()
            latents_df = pd.concat([latents_df, pd.DataFrame(latents_numpy)], ignore_index=True)
        if remainder:
            image_remainder = torch.tensor(modis_data[-remainder:])
            image_tensor = torch.tensor(image_remainder)
            latents_tensor = model(image_tensor)
            if using_gpu:
                latents_numpy = latents_tensor.to_cpu().numpy()
            else:
                latents_numpy = latents_tensor.numpy()
            latents_df = pd.concat([latents_df, pd.DataFrame(latents_numpy)], ignore_index=True)
            latents_numpy = latents_df.values
    with h5py.File(save_path, 'w') as file:
        file.create_dataset(save_key, data=latents_numpy)
    print("Wrote: {}".format(save_path))
        
if __name__ == "__main__":
    
    model_path = "./experiments/base_modis_model/SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_1_cosine_warm/" \
             "SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm"
    model_name = "last.pth"
    
    opt_path = './experiments/base_modis_model/opts.json'
    opt = Params(opt_path)
    opt = option_preprocess(opt)
    
    modis_dataset_path = "./experiments/modis_dataset/MODIS_2010_95clear_128x128_inpaintT_preproc_0.8valid.h5"
    
    with h5py.File(modis_dataset_path, 'r') as file:
        dataset_train = file['train'][:]
        
    save_path = './experiments/modis_latents/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    model_path_title = os.path.join(model_path, model_name)
    model_name = model_name.split('.')[0]
    latents_path = os.path.join(save_path, f'modis_latents_{model_name}.h5')
    save_key = 'modis_latents'
    
    model_latents_extract(opt, dataset_train, model_path_title, latents_path, save_key)