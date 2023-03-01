""" Simple routines to work on a single image """
import os
from pkg_resources import resource_filename
import numpy as np

import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms

from ulmo.ssl import train_util
from ulmo.ssl.util import TwoCropTransform
from ulmo import io as ulmo_io

class BatchDataset(Dataset):
    """ Class to generate a batch of images
    from the input image and the pp_file
    """
    def __init__(self, image, batch_size:int, pp_file:str):
        self.batch_size = batch_size
        # Load up a faux batch
        pp_hf = h5py.File(f, 'r')
        random_idx  = np.random.randint(
            0, pp_hf['valid'].shape[0], batch_size)
        self.images = pp_hf['valid'][random_idx, ...]
        # Set ours of interest
        self.images[0] = np.resize(image,
                                   (1, image.shape[-1], image.shape[-1]))

    def __len__(self):
        return self.batch_size

    def __getitem__(self, global_idx):     
        data = self.images[global_idx]
        # For SSL
        data = np.resize(data, (1, data.shape[-1], data.shape[-1]))
        data = np.repeat(data, 3, axis=0)
        
        # Metadata
        metadata = None
        # Return
        return data, metadata

    
class ImageDataset(Dataset):
    def __init__(self, image, transform):
        self.transform = transform
        self.images = [image]

    def __len__(self):
        return 1

    def __getitem__(self, global_idx):     
        image = self.images[global_idx]
        image_transposed = np.transpose(image, (1, 2, 0))
        image_transformed = self.transform(image_transposed)
        
        return image_transformed

def image_loader(image, version='v4'):
    if version == 'v3':
        transforms_compose = transforms.Compose(
            [train_util.RandomRotate(verbose=True), 
            train_util.JitterCrop(verbose=True), 
            train_util.GaussianNoise(), 
            transforms.ToTensor()])
    elif version == 'v4':
        opt_file = os.path.join(resource_filename('ulmo', 'runs'),
            'SSL', 'MODIS', 'v4', 'opts_ssl_modis_v4.json')
        # loading parameters json file
        opt = ulmo_io.Params(opt_file)
        opt = train_util.option_preprocess(opt)
        transforms_compose = transforms.Compose(
            [train_util.RandomFlip(verbose=True), 
             train_util.RandomRotate(verbose=True),
             train_util.JitterCrop(crop_dim=opt.random_jitter[0],
                                       jitter_lim=opt.random_jitter[1],
                                       rescale=0, verbose=True),
             train_util.Demean(), 
             train_util.ThreeChannel(), 
             transforms.ToTensor()])
    else:
        raise IOError("Not ready for this version: {}".format(version))

    
    image_dataset = ImageDataset(
        image, transform=TwoCropTransform(
            transforms_compose))
    train_loader = torch.utils.data.DataLoader(
                    image_dataset, batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False, sampler=None)
    
    return train_loader
    