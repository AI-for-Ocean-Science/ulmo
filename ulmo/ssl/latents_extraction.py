from __future__ import print_function

import time
import os
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import trange

import torch
import tqdm

from ulmo.utils import HDF5Dataset, id_collate

from ulmo.ssl.train_util import Params, option_preprocess
from ulmo.ssl.train_util import modis_loader, set_model
from ulmo.ssl.train_util import train_model

from IPython import embed


class HDF5RGBDataset(torch.utils.data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Parameters:
        file_path: Path to the HDF5 file.
        dataset_names: List of dataset names to gather. 
        
    Objects will be returned in this order.
    """
    def __init__(self, file_path, partition):
        super().__init__()
        self.file_path = file_path
        self.partition = partition
        self.meta_dset = partition + '_metadata'
        # s3 is too risky and slow here
        self.h5f = h5py.File(file_path, 'r')

    def __len__(self):
        #return 100  # Debuggin
        return self.h5f[self.partition].shape[0]
    
    def __getitem__(self, index):
        data = self.h5f[self.partition][index]
        data = np.resize(data, (1, data.shape[-1], data.shape[-1]))
        data = np.repeat(data, 3, axis=0)
        #if self.meta_dset in self.h5f.keys():
        #    metadata = self.h5f[self.meta_dset][index]
        #else:
        metadata = None
        return data, metadata
    

def build_loader(data_file, dataset, batch_size=1, num_workers=1):
    # Generate dataset
    """
    This function is used to create the data loader for the latents
    creating (evaluation) process.
    Args: 
        data_file: (str) path of data file.
        dataset: (str) key of the used data in data_file.
        batch_size: (int) batch size of the evalution process.
        num_workers: (int) number of workers used in loading data.
    
    Returns:
        dset: (HDF5RGBDataset) HDF5 dataset of data_file.
        loader: (torch.utils.data.Dataloader) Dataloader created 
            using data_file.
    """
    
    dset = HDF5RGBDataset(data_file, partition=dataset)

    # Generate DataLoader
    loader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=False, 
        collate_fn=id_collate,
        drop_last=False, num_workers=num_workers)
    
    return dset, loader

def calc_latent(model, image_tensor, using_gpu):
    """
    This is a function to calculate the latents.
    Args:
        model: (SupConResNet) model class used for latents.
        image_tensor: (torch.tensor) image tensor of the data set.
        using_gpu: (bool) flag for cude usage.
    """
    model.eval()
    if using_gpu:
        latents_tensor = model(image_tensor.cuda())
        latents_numpy = latents_tensor.cpu().numpy()
    else:
        latents_tensor = model(image_tensor)
        latents_numpy = latents_tensor.numpy()
    return latents_numpy


def model_latents_extract(opt, modis_data_file, modis_partition, 
                          model_path, save_path, 
                          save_key,
                          remove_module=True):
    """
    This function is used to obtain the latents of the training data.
    Args:
        opt: (Parameters) parameters used to create the model.
        modis_data_file: (str) path of modis_data_file.
        modis_partition: (str) key of the h5py file [e.g. 'train', 'valid'].
        model_path: (string) path of the saved model file.
        save_path: (string) path for saving the latents.
        save_key: (string) path for the key of the saved latents.
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

    # Data
    _, loader = build_loader(modis_data_file, modis_partition)

    print("Beginning to evaluate")
    model.eval()
    with torch.no_grad():
        latents_numpy = [calc_latent(
            model, data[0], using_gpu) for data in tqdm.tqdm(
                loader, total=len(loader), unit='batch', 
                desc='Computing latents')]
    
    with h5py.File(save_path, 'w') as file:
        file.create_dataset(save_key, data=np.concatenate(latents_numpy))
    print("Wrote: {}".format(save_path))

    

def orig_latents_extract(opt, modis_data, 
                          model_path, save_path, 
                          save_key,
                          remove_module=True):
    """
    This function is used to obtain the latents of the training data.
    Args:
        opt: (Parameters) parameters used to create the model
        modis_data_file: (str) 
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
    model.eval()
    with torch.no_grad():
        for i in trange(num_steps):
            image_batch = modis_data[i*batch_size: (i+1)*batch_size]
            image_tensor = torch.tensor(image_batch)
            if using_gpu:
                latents_tensor = model(image_tensor.cuda())
                latents_numpy = latents_tensor.cpu().numpy()
            else:
                latents_tensor = model(image_tensor)
                latents_numpy = latents_tensor.numpy()
            latents_df = pd.concat([latents_df, pd.DataFrame(latents_numpy)], ignore_index=True)
        if remainder:
            image_remainder = torch.tensor(modis_data[-remainder:])
            image_tensor = torch.tensor(image_remainder)
            if using_gpu:
                latents_tensor = model(image_tensor.cuda())
                latents_numpy = latents_tensor.cpu().numpy()
            else:
                latents_tensor = model(image_tensor)
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