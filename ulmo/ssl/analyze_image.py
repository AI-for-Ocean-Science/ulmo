""" Package to explore an input image """
import os
import numpy as np

import torch

from ulmo.ssl import latents_extraction
from ulmo import io as ulmo_io

def get_latents(img:np.ndarray, 
                model_file:str, 
                opt:ulmo_io.Params):

    # Build the SSL model
    model_base = os.path.basename(model_file)
    if not os.path.isfile(model_base):
        ulmo_io.download_file_from_s3(model_base, model_file)
    else:
        print(f"Using already downloaded {model_base} for the model")

    # DataLoader
    dset = torch.utils.data.TensorDataset(torch.from_numpy(img).float())
    data_loader = torch.utils.data.DataLoader(
        dset, batch_size=1, shuffle=False, collate_fn=None,
        drop_last=False, num_workers=1)

    # Time to run
    latents = latents_extraction.model_latents_extract(
        opt, 'None', 'valid', 
        model_base, None, None,
         loader=data_loader)

    # Return
    return latents

def calc_DT40(images, random_jitter:list,
              verbose=False, debug=False):
    """Calculate DT40 for a given image or set of images

    Args:
        images (np.ndarray): 
            Analyzed shape is (N, 64, 64)
            but a variety of shapes is allowed
        random_jitter (list):
            range to crop, amount to randomly jitter
    Returns:
        np.ndarray or float: DT40
    """
    if verbose:
        print("Calculating T90")
    # If single image, reshape into fields
    single = False
    if len(images.shape) == 4:
        fields = images[...,0,...]
    elif len(images.shape) == 2:
        fields = np.expand_dims(images, axis=0) 
        single = True
    elif len(images.shape) == 3:
        fields = images
    else:
        raise IOError("Bad shape for images")

    # Center
    xcen = fields.shape[-2]//2    
    ycen = fields.shape[-1]//2    
    dx = random_jitter[0]//2
    dy = random_jitter[0]//2
    if verbose:
        print(xcen, ycen, dx, dy)
    
    T_90 = np.percentile(fields[..., xcen-dx:xcen+dx,
        ycen-dy:ycen+dy], 90., axis=(1,2))
    if verbose:
        print("Calculating T10")
    T_10 = np.percentile(fields[..., xcen-dx:xcen+dx,
        ycen-dy:ycen+dy], 10., axis=(1,2))
    #T_10 = np.percentile(fields[:, 0, 32-20:32+20, 32-20:32+20], 
    #    10., axis=(1,2))
    DT_40 = T_90 - T_10

    # Return
    if single:
        return DT_40[0]
    else:
        return DT_40