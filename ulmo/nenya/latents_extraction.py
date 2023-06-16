from __future__ import print_function

import time
import os
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import trange

import torch
import tqdm

from ulmo.io import Params
from ulmo.utils import id_collate

from ulmo.nenya.train_util import option_preprocess
from ulmo.nenya.train_util import modis_loader, set_model

from IPython import embed


class HDF5RGBDataset(torch.utils.data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Parameters:
        file_path: Path to the HDF5 file.
        dataset_names: List of dataset names to gather. 
        allowe_indices (np.ndarray): Set of images that can be grabbed
        
    Objects will be returned in this order.
    """
    def __init__(self, file_path, partition, allowed_indices=None):
        super().__init__()
        self.file_path = file_path
        self.partition = partition
        self.meta_dset = partition + '_metadata'
        # s3 is too risky and slow here
        self.h5f = h5py.File(file_path, 'r')
        # Indices -- allows us to work on a subset of the images by indices
        self.allowed_indices = allowed_indices
        if self.allowed_indices is None:
            self.allowed_indices = np.arange(self.h5f[self.partition].shape[0])

    def __len__(self):
        return self.allowed_indices.size
        #return self.h5f[self.partition].shape[0]
    
    def __getitem__(self, index):
        # Grab it
        data = self.h5f[self.partition][self.allowed_indices[index]]
        # Resize
        data = np.resize(data, (1, data.shape[-1], data.shape[-1]))
        data = np.repeat(data, 3, axis=0)
        # Metadata
        metadata = None
        return data, metadata


def build_loader(data_file, dataset, batch_size=1, num_workers=1,
                 allowed_indices=None):
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
    dset = HDF5RGBDataset(data_file, partition=dataset, allowed_indices=allowed_indices)

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
                          model_path, save_path, save_key,
                          remove_module=True, loader=None,
                          allowed_indices=None):
    """
    This function is used to obtain the latents of input data.
    
    Args:
        opt: (Parameters) parameters used to create the model.
        modis_data_file: (str) path of modis_data_file.
        modis_partition: (str) key of the h5py file [e.g. 'train', 'valid'].
        model_path: (string) path of the saved model file.
        save_path: (string or None) path for saving the latents.
        save_key: (string or None) path for the key of the saved latents.
        loader: (torch.utils.data.DataLoader, optional) Use this DataLoader, if provided
        save_path: (str or None) path for saving the latents.
        save_key: (str or None) path for the key of the saved latents.

    Returns:
        np.ndarray: latents_numpy
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

    # TODO -- Get this right on the ssl_full branch-- 
    # Create Data Loader for evaluation
    #batch_size_eval, num_workers_eval = opt.batch_size_eval, opt.num_workers_eval
    batch_size_eval, num_workers_eval = opt.batch_size_valid, opt.num_workers

    # Data
    if loader is None:
        _, loader = build_loader(modis_data_file, modis_partition, 
                                 batch_size_eval, num_workers_eval,
                                 allowed_indices=allowed_indices)

    print("Beginning to evaluate")
    model.eval()
    with torch.no_grad():
        latents_numpy = [calc_latent(
            model, data[0], using_gpu) for data in tqdm.tqdm(
                loader, total=len(loader), unit='batch', 
                desc='Computing latents')]
    
    # Save
    if save_path is not None:
        with h5py.File(save_path, 'w') as file:
            file.create_dataset(save_key, data=np.concatenate(latents_numpy))
        print("Wrote: {}".format(save_path))

    return np.concatenate(latents_numpy)
    
