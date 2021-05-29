import os
import sys
import numpy as np
import math
import time

import json
import skimage.transform
import h5py

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

from ulmo.ssl.resnet_big import SupConResNet
from ulmo.ssl.losses import SupConLoss

from ulmo.ssl.util import TwoCropTransform, AverageMeter
from ulmo.ssl.util import warmup_learning_rate

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    This module comes from:
    https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
    
def option_preprocess(opt: Params):
    """
    Args:
        opt: (Params) json used to store the training hyper-parameters
    Returns:
        opt: (Params) processed opt
    """

    # check if dataset is path that passed required arguments
    if opt.modis_data == True:
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './experimens/datasets/'
    opt.model_path = f'./experiments/{opt.method}/{opt.dataset}_models'
    opt.tb_path = f'./experiments/{opt.method}/{opt.dataset}_tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt
    
class RandomRotate:
    def __call__(self, image):
    # print("RR", image.shape, image.dtype)
        rang = np.float32(360*np.random.rand(1))
        print('random angle = {}'.format(rang))
        return (skimage.transform.rotate(image, rang)).astype(np.float32)
        #return (skimage.transform.rotate(image, np.float32(360*np.random.rand(1)))).astype(np.float32)
    
class JitterCrop:
    def __init__(self, crop_dim=32, rescale=2, jitter_lim=0):
        self.crop_dim = crop_dim
        self.offset = self.crop_dim//2
        self.jitter_lim = jitter_lim
        self.rescale = rescale
        
    def __call__(self, image):
        center_x = image.shape[0]//2
        center_y = image.shape[0]//2
        if self.jitter_lim:
            center_x += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
            center_y += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))

        image_cropped = image[(center_x-self.offset):(center_x+self.offset), (center_y-self.offset):(center_y+self.offset), 0]
        image = np.expand_dims(skimage.transform.rescale(image_cropped, self.rescale), axis=-1)
        image = np.repeat(image, 3, axis=-1)
        
        return image
    
class GaussianNoise:
    def __init__(self, instrument_noise=(0, 0.1)):
        self.noise_mean = instrument_noise[0]
        self.noise_std = instrument_noise[1]

    def __call__(self, image):
        noise_shape = image.shape
        noise = np.random.normal(self.noise_mean, self.noise_std, size=noise_shape)
        image += noise

        return image
    
class ModisDataset(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform

    def _open_file(self):
        self.files = h5py.File(self.data_path, 'r')

    def __len__(self):
        self._open_file()
        num_samples = self.files['train'].shape[0]
        return num_samples

    def __getitem__(self, global_idx):     
        self._open_file()
        image = self.files['train'][global_idx]
        image_transposed = np.transpose(image, (1, 2, 0))
        image_transformed = self.transform(image_transposed)
        
        return image_transformed
    
def modis_loader(opt):
    transforms_compose = transforms.Compose([RandomRotate(),
                                             JitterCrop(),
                                             GaussianNoise(),
                                             transforms.ToTensor()])
    
    modis_path = opt.data_folder
    modis_dataset = ModisDataset(modis_path, transform=TwoCropTransform(transforms_compose))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
                    modis_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                    num_workers=opt.num_workers, pin_memory=False, sampler=train_sampler)
    
    return train_loader
    
def set_model(opt, cuda_use=True): 
    model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)
    
    if torch.cuda.is_available() and cuda_use:
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def train_modis(train_loader, model, criterion, optimizer, epoch, opt, cuda_use=True):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, images in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available() and cuda_use:
            images = images.cuda(non_blocking=True)
            #labels = labels.cuda(non_blocking=True)
        bsz = images.shape[0] // 2

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

    
