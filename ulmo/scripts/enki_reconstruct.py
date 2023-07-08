""" Script to run Enki for image reconstructions """

import os
import argparse
from pathlib import Path

import h5py
import torch

import timm.optim.optim_factory as optim_factory

from ulmo.utils import HDF5Dataset, id_collate

from ulmo.mae import models_mae
from ulmo.mae import reconstruct
import ulmo.mae.util.misc as misc
from ulmo.mae.util.hdfstore import HDF5Store
from ulmo.mae.enki_utils import img_filename, mask_filename
from ulmo.mae.reconstruct import reconstruct_one_epoch
from ulmo.mae.util.misc import NativeScalerWithGradNormCount as NativeScaler

from ulmo import io as ulmo_io

from IPython import embed

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
    parser.add_argument('--upload_path', type=str, help='s3 path for uploading the reconstructed file')
    parser.add_argument('--mask_upload_path', type=str, help='s3 path for uploading the mask file')
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
    parser.add_argument('--use_masks', action='store_true',
                        help='Use Masks?')
    parser.add_argument('--debug', action='store_true',
                        help='Debug')

    return parser

def main(args):

    # Handle distributed
    misc.init_distributed_mode(args)
    
    # Load data (locally)
    dataset_train = HDF5Dataset(args.data_path, 
                                partition='valid',
                                return_mask=args.use_masks)
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
    if args.upload_path is None:
        upload_path = img_filename(int(100*args.model_training_mask), int(100*args.mask_ratio))
    else:
        upload_path = args.upload_path
        
    filepath = os.path.join(args.output_dir, os.path.basename(upload_path))
    images = HDF5Store(filepath, 'valid', shape=dshape)
    
    # set up mask file and upload path
    if args.mask_upload_path is None:
        mask_upload_path = mask_filename(int(100*args.model_training_mask), int(100*args.mask_ratio))
    else:
        mask_upload_path = args.mask_upload_path
    mask_filepath = os.path.join(args.output_dir, os.path.basename(mask_upload_path))
    masks = HDF5Store(mask_filepath, 'valid', shape=dshape)
    
    print(f"Saving to file {filepath} and uploading to {upload_path}")
    
    print(f"Start reconstructing for {data_length} images")

    if args.distributed: # args.distributed:??
        raise ValueError("Distributed not supported")
        data_loader_train.sampler.set_epoch(epoch)

    # Do one epoch of reconstruction
    #  This runs on all the data except the last batch
    #  which was dropped (in case it was the wrong length)
    train_stats = reconstruct_one_epoch(
        model, data_loader_train,
        optimizer, device, loss_scaler,
        args.mask_ratio, args.batch_size, args.accum_iter,
        image_store=images,
        mask_store=masks,
        use_mask=args.use_masks
    )

    # Now get the last batch and write to disk
    print("Reconstructing the batch remainder")
    reconstruct.run_remainder(model, data_length, 
                              images, masks,
                              args.batch_size, 
                              args.data_path,
                              args.mask_ratio) 
    ulmo_io.upload_file_to_s3(filepath, upload_path)
    ulmo_io.upload_file_to_s3(mask_filepath, mask_upload_path)
    


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    
# On Jupyter
# cp ulmo/mae/correct_helpers.py /opt/conda/lib/python3.10/site-packages/timm-0.3.2-py3.10.egg/timm/models/layers/helpers.py
# python /home/jovyan/Oceanography/python/ulmo/ulmo/scripts/enki_reconstruct.py --mask_ratio 0.1 --data_path VIIRS_all_100clear_preproc.h5 --output_dir output --resume checkpoint-270.pth --upload_path s3://llc/mae/Recon/VIIRS_100clear_t10_p10.h5 --mask_upload_path s3://llc/mae/Recon/VIIRS_100clear_t10_p10_mask.h5