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
import math
import numpy as np
import h5py
from typing import Iterable

import ulmo.mae.util.misc as misc
from ulmo.mae import models_mae
import ulmo.mae.util.lr_sched as lr_sched
from ulmo.mae.util.hdfstore import HDF5Store

from IPython import embed

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
    """ Run Enki on the last batch of data

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
            run_one_image(img, model, mask_ratio, 
                          image_store=image_store, 
                          mask_store=mask_store)


def reconstruct_one_epoch(model: torch.nn.Module,
                         data_loader: Iterable, 
                         optimizer: torch.optim.Optimizer,
                         device: torch.device, loss_scaler,
                         mask_ratio:float,
                         batch_size:int,
                         accum_iter:int,
                         image_store:HDF5Store=None,
                         mask_store:HDF5Store=None,
                         log_writer=None,
                         use_mask:bool=False):
    """ Reconstruct a single epoch of data, modulo
    the very last bit (in case it does not match the batch_size)

    Args:
        model (torch.nn.Module): MAE model
        data_loader (Iterable): torch DataLoader
        optimizer (torch.optim.Optimizer): _description_
        device (torch.device): _description_
        loss_scaler (_type_): _description_
        mask_ratio (float): _description_
        batch_size (int): _description_
        accum_iter (int): _description_
        image_store (HDF5Store, optional): _description_. Defaults to None.
        mask_store (HDF5Store, optional): _description_. Defaults to None.
        log_writer (_type_, optional): _description_. Defaults to None.
        use_mask (bool, optional): If True, use the user-supplied mask
            and not a random one

    Returns:
        dict: _description_
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Reconstructing:'
    print_freq = 20
    
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
#    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, items in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # Unpack
        samples = items[0]
        samples = samples.to(device, non_blocking=True)

        #embed(header='reconstruct_one_epoch 196')
        if use_mask:
            masks = items[1]
            #masks = torch.Tensor(masks).to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            loss, y, mask = model(samples, mask_ratio=mask_ratio,
                                  masks=masks if use_mask else None)
            
        ## --------------------- New stuff -----------------------
        # note: despite leaving the setup for it this is not DDP comptible yet

        # unpatchify y
        y = model.unpatchify(y)
        #y = y.detach()  # nchw (# images, channels, height, width)
              
        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *1)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        #mask = mask.detach()  # nchw (# images, channels, height, width)

        im_masked = samples * (1 - mask)
        im_paste = samples * (1 - mask) + y * mask
        im = im_paste.cpu().detach().numpy()
        m = mask.cpu().detach().numpy()
        m = np.squeeze(m, axis=1)
        for i in range(batch_size):
            image_store.append(im[i])
            mask_store.append(m[i])
        
        # --------------------------------------------------------
        
        # just extra. Leaving this in case removing it breaks it
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}