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



'''
def deprecated_prepare_model(args):
    """

    Args:
        args (_type_): _description_
            args.model -- Specifies the name of the model to use

    Returns:
        _type_: _description_
    """
    # build model
    device = torch.device(args.device)
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.to(device)
    model_without_ddp = model
    
    if args.distributed: # args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
        model_without_ddp = model.module
    
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    return model, optimizer, device, loss_scaler


def orig_run_one_image(img:np.ndarray, model, mask_ratio:float, file:str, mask_file:str):
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

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches (image of interest)
    im_paste = x * (1 - mask) + y * mask
    temp = im_paste.cpu().detach().numpy()
    #from IPython import embed; embed(header='225 of extract')
    im = np.squeeze(temp, axis=3)
    m = mask.cpu().detach().numpy()
    m = np.squeeze(m, axis=3)
    m = np.squeeze(m, axis=0)
    # TODO: check mask size
    
    file.append(im)
    mask_file.append(m)
'''


def run_one_image(img:np.ndarray, model, mask_ratio:float):
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
    
def run_remainder(model, data_length, file, mask_file:str, 
                  batch_size:int, data_path:str, 
                  mask_ratio:float, imsize:int=64):
    # Indices to run on
    start = (data_length // batch_size) * batch_size
    end = data_length
    # Do it!
    with h5py.File(data_path, 'r') as f:
        for i in range(start, end):
            img = f['valid'][i][0]
            img.resize((imsize,imsize,1))
            assert img.shape == (imsize, imsize, 1)
            run_one_image(img, model, mask_ratio, file, mask_file)
