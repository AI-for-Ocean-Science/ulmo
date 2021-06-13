from __future__ import print_function

import time
import os
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import trange

import torch

from util import adjust_learning_rate
from util import set_optimizer, save_model

from my_util import Params, option_preprocess
from my_util import modis_loader, set_model
from my_util import train_modis

def model_latents_extract(opt, modis_data, model_path, save_path, save_key):
    """
    This function is used to obtain the latents of the training data.
    Args:
        opt: (Parameters) parameters used to create the model.
        modis_data: (numpy.array) modis_data used in the latents
            extraction process.
        model_path: (string) path of the model file. 
        save_path: (string) path to save the extracted latents
        save_key: (string) key of the h5py file for the latents.
    """
    model, _ = set_model(opt, cuda_use=True)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['model'])
    modis_data = np.repeat(modis_data, 3, axis=1)
    num_samples = modis_data.shape[0]
    batch_size = opt.batch_size
    #batch_size = 1
    num_steps = num_samples // batch_size
    #num_steps = 1
    remainder = num_samples % batch_size
    latents_df = pd.DataFrame()
    with torch.no_grad():
        for i in trange(num_steps):
            image_batch = modis_data[i*batch_size: (i+1)*batch_size]
            image_tensor = torch.tensor(image_batch)
            latents_tensor = model(image_tensor)
            latents_numpy = latents_tensor.cpu().numpy()
            latents_df = pd.concat([latents_df, pd.DataFrame(latents_numpy)], ignore_index=True)
        if remainder:
            image_remainder = modis_data[-remainder:]
            image_tensor = torch.tensor(image_remainder)
            latents_tensor = model(image_tensor)
            latents_numpy = latents_tensor.cpu().numpy()
            latents_df = pd.concat([latents_df, pd.DataFrame(latents_numpy)], ignore_index=True)
            latents_numpy = latents_df.values
    with h5py.File(save_path, 'a') as file:
        file.create_dataset(save_key, data=latents_numpy)
        
if __name__ == "__main__":
    
    model_path = "./experiments/SimCLR/modis_models_v2/SimCLR_modis_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_3_R2019_2010_cosine_warm/"
    model_name_list = ["ckpt_epoch_5.pth", "ckpt_epoch_10.pth", "ckpt_epoch_15.pth", "last.pth"]
    
    opt_path = './experiments/modis_model_v2/opts.json'
    opt = Params(opt_path)
    opt = option_preprocess(opt)
    
    modis_dataset_path = "./experiments/datasets/modis_dataset/MODIS_R2019_2010_95clear_128x128_preproc_std.h5"
    
    with h5py.File(modis_dataset_path, 'r') as file:
        dataset_train = file['train'][:]
        dataset_valid = file['valid'][:]
        
    save_path = './experiments/modis_latents_v2/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    save_key_train = 'modis_latents_v2_train'
    save_key_valid = 'modis_latents_v2_valid'
    
    for model_name in model_name_list:
        model_path_title = os.path.join(model_path, model_name)
        model_name = model_name.split('.')[0]
        latents_path = os.path.join(save_path, f'modis_R2019_2010_latents_{model_name}_v2.h5')  
        model_latents_extract(opt, dataset_train, model_path_title, latents_path, save_key_train)
        model_latents_extract(opt, dataset_valid, model_path_title, latents_path, save_key_valid)
