#!/usr/bin/env python
# coding: utf-8

# ## Masked Autoencoders: Visualization Demo
# 
# This is a visualization demo using our pre-trained MAE models. No GPU is needed.

# ### Prepare
# Check environment. Install packages if in Colab.
# 

# In[1]:


import sys
import os
import requests

import torch
import numpy as np
import h5py

import matplotlib.pyplot as plt
from PIL import Image
from ulmo.plotting import plotting
from hdfstore import HDF5Store

# check whether run in Colab
if 'google.colab' in sys.modules:
    print('Running in Colab.')
    get_ipython().system('pip3 install timm==0.4.5  # 0.3.2 does not work in Colab')
    get_ipython().system('git clone https://github.com/facebookresearch/mae.git')
    sys.path.append('./mae')
else:
    sys.path.append('..')
import models_mae


# ### Define utils

# In[2]:


# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 1
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_LLC_patch4'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    #checkpoint = torch.load(chkpt_dir, map_location='cpu')
    checkpoint = torch.load(chkpt_dir)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, mask_ratio):
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

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches (image of interest)
    im_paste = x * (1 - mask) + y * mask
    temp = im_paste.cpu().detach().numpy()
    #from IPython import embed; embed(header='225 of extract')
    im = np.squeeze(temp, axis=3)
    
    file = HDF5Store('reconstructions.h5', 'valid', shape=im.shape)
    file.append(im)



# Load model
chkpt_dir = 'checkpoint-200.pth'
#chkpt_dir = 'checkpoint-100.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_LLC_patch4')
print('Model loaded.')


# make random mask reproducible (comment out to make it change)
#torch.manual_seed(2)

print('running')
# Run MAE
filepath = '../LLC_uniform144_test_preproc.h5'
with h5py.File(filepath, 'r') as f:
    len_valid = f['valid'].shape[0]
    for i in range(10):
        if i%100 == 0:
            print('Reconstructing image ', i)
        img = f['valid'][i][0]
        img.resize((64,64,1))
        
        assert img.shape == (64, 64, 1)
        run_one_image(img, model_mae, 0.10)
    
    





