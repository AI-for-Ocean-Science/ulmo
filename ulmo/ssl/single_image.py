""" Simple routines to work on a single image """
import os
from pkg_resources import resource_filename
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ulmo.ssl import train_util
from ulmo.ssl.util import TwoCropTransform
from ulmo import io as ulmo_io
    
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
            [train_util.RandomRotate(), 
            train_util.JitterCrop(), 
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
    