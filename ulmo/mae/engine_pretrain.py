# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import numpy as np

import torch

import ulmo.mae.util.misc as misc
import ulmo.mae.util.lr_sched as lr_sched
from ulmo.mae.util.hdfstore import HDF5Store


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        
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
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def reconstruct_one_epoch(model: torch.nn.Module,
                         data_loader: Iterable, 
                         optimizer: torch.optim.Optimizer,
                         device: torch.device, loss_scaler,
                         mask_ratio:float,
                         batch_size:int,
                         accum_iter:int,
                         image_store:HDF5Store=None,
                         mask_store:HDF5Store=None,
                         log_writer=None):
    """ Reconstruct a single epoch

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
    
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        samples = samples.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            loss, y, mask = model(samples, mask_ratio=mask_ratio)
            
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