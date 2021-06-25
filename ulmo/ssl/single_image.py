""" Simple routines to work on a single image """
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ulmo.ssl import my_util
from ulmo.ssl.util import TwoCropTransform
    
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

def image_loader(image):
    transforms_compose = transforms.Compose(
        [my_util.RandomRotate(), 
         my_util.JitterCrop(), 
         my_util.GaussianNoise(), 
         transforms.ToTensor()])
    
    image_dataset = ImageDataset(
        image, transform=TwoCropTransform(
            transforms_compose))
    train_loader = torch.utils.data.DataLoader(
                    image_dataset, batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False, sampler=None)
    
    return train_loader
    