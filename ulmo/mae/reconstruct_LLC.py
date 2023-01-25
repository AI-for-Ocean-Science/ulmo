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
import argparse
from pathlib import Path

import torch
import numpy as np
import h5py
import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import matplotlib.pyplot as plt
from PIL import Image
from util.hdfstore import HDF5Store
import util.misc as misc
from ulmo import io as ulmo_io
from ulmo.utils import HDF5Dataset, id_collate, get_quantiles

import models_mae
from mae_utils import img_filename
from engine_pretrain import reconstruct_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('MAE reconstruction', add_help=False)
    
    # Model parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--model', default='mae_vit_LLC_patch4', type=str, metavar='MODEL',
                        help='Name of model to use')
    parser.add_argument('--input_size', default=64, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.10, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--model_training_mask', default=0.10, type=float,
                        help='Masking ratio the model trained with')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Dataset parameters
    parser.add_argument('--data_path', default='LLC_uniform144_nonoise_preproc.h5', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='checkpoint to load')
    
    # ????? just in case it's important
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True) # set to false if getting cuda error

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def prepare_model(args):
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


def main(args):
    misc.init_distributed_mode(args)
    
    # load data
    dataset_train = HDF5Dataset(args.data_path, partition='valid')
    with h5py.File(args.data_path, 'r') as f:
        dshape=f['valid'][0].shape
        
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=id_collate,
        drop_last=True,
    )
    data_length = len(data_loader_train.dataset)
    print("training",  data_length)
    print("Datasets loaded")
    
    # setup model and parameters
    model, optimizer, device, loss_scaler = prepare_model(args)
    print(args.model, "loaded")
    
    # set up file and upload path
    upload_path = img_filename(int(100*args.model_training_mask), int(100*args.mask_ratio))
    filepath = os.path.join(args.output_dir, os.path.basename(upload_path))
    file = HDF5Store(filepath, 'valid', shape=dshape)
    print(f"Saving to file {filepath} and uploading to {upload_path}")
    
    print(f"Start reconstructing for {data_length} images")

    if args.distributed: # args.distributed:??
        data_loader_train.sampler.set_epoch(epoch)
    train_stats = reconstruct_one_epoch(
        model, data_loader_train,
        optimizer, device, loss_scaler,
        file=file,
        args=args
    )
    ulmo_io.upload_file_to_s3(filepath, upload_path)
    
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    
  
