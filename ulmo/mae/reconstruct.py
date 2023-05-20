#!/usr/bin/env python
# coding: utf-8

# ## Masked Autoencoders: Visualization Demo
# 
# This is a visualization demo using our pre-trained MAE models. No GPU is needed.

# ### Prepare
# Check environment. Install packages if in Colab.
# 

# In[1]:

import torch
import numpy as np
import h5py

import ulmo.mae.util.misc as misc
from ulmo.mae import models_mae
from ulmo.mae.util.hdfstore import HDF5Store

def run_one_image(img:np.ndarray, model, mask_ratio:float, 
                  image_store:HDF5Store=None, mask_store:HDF5Store=None):
    """ Reconstruct a single image with a MAE model

    Args:
        img (np.ndarray): 
            Image with shape (imsize, imsize, 1)
        model (_type_): _description_
            Enki model
        mask_ratio (float): 
            Patch masking ratio, e.g. 0.1 means 10% of patches are masked
        image_store (HDF5Store, optional): 
            Object to store the reconstructed image
        mask_store (HDF5Store, optional): 
            Object to store the mask image
    """
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = x.cuda()
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *1)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach()
    
    x = torch.einsum('nchw->nhwc', x)

    # Store?
    if image_store is not None or mask_store is not None:
        # MAE reconstruction pasted with visible patches (image of interest)
        im_paste = x * (1 - mask) + y * mask
        temp = im_paste.cpu().detach().numpy()
        #from IPython import embed; embed(header='225 of extract')
        im = np.squeeze(temp, axis=3)
        m = mask.cpu().detach().numpy()
        m = np.squeeze(m, axis=3)
        m = np.squeeze(m, axis=0)
        # TODO: check mask size
        
        # Save
        image_store.append(im)
        mask_store.append(m)


def x_run_one_image(img:np.ndarray, model, mask_ratio:float):
    """ Reconstruct a single image

    Args:
        img (np.ndarray): _description_
        model (_type_): _description_
        mask_ratio (float): 
            Patch fraction, e.g. 0.3 means insert patches on 30% of the image

    Returns:
        tuple: mask, reconstructed image, original image
    """
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *1)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)
    return mask, x, y
    

def run_remainder(model, data_length:int, 
                  image_store:HDF5Store, mask_store:HDF5Store,
                  batch_size:int, data_path:str, 
                  mask_ratio:float, imsize:int=64):
    """ Run Enki on a set of data

    Args:
        model (_type_): _description_
        data_length (int): _description_
        image_store (HDF5Store): 
            Stores the reconstructed images
        mask_store (HDF5Store): 
            Stores the mask images
        batch_size (int): _description_
        data_path (str): _description_
        mask_ratio (float): _description_
        imsize (int, optional): _description_. Defaults to 64.
    """
    # Indices to run on
    start = (data_length // batch_size) * batch_size
    end = data_length
    # Do it!
    with h5py.File(data_path, 'r') as f:
        for i in range(start, end):
            img = f['valid'][i][0]
            img.resize((imsize,imsize,1))
            #assert img.shape == (imsize, imsize, 1)
            run_one_image(img, model, mask_ratio, image_store=image_store, mask_store=mask_store)
