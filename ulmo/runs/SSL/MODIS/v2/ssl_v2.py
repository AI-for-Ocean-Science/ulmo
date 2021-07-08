from __future__ import print_function

import sys
target_path = '/ulmo/'
sys.path.append(target_path)

import time
import os
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import trange
import argparse

import torch

from ulmo.ssl.util import adjust_learning_rate
from ulmo.ssl.util import set_optimizer, save_model

from ulmo.ssl.train_util import Params, option_preprocess
from ulmo.ssl.train_util import modis_loader, set_model
from ulmo.ssl.train_util import train_model
from ulmo.ssl import train_modis

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("argument for training.")
    parser.add_argument("--opt_path", type=str, help="path of 'opt.json' file.")
    parser.add_argument("--func_flag", type=str, help="flag of the function to be execute: 'train' or 'evaluate'.")
    args = parser.parse_args()
    
    return args

def main_train(opt_path: str):
    # loading parameters json file
    opt = Params(opt_path)
    opt = option_preprocess(opt)

    # build data loader
    train_loader = modis_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt, cuda_use=opt.cuda_use)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    # training routine
    for epoch in trange(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train_model(train_loader, model, criterion, optimizer, epoch, opt, cuda_use=opt.cuda_use)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

def model_latents_extract(opt, modis_data, model_path, save_path, save_key):
    """
    This function is used to obtain the latents of the training data.
    Args:
        opt: (Parameters) parameters used to create the model.
        modis_data: (numpy.array) modis_data used in the latents
            extraction process.
        model_path: (string) path of the model file. 
        save_path: (string) path to save the extracted latents
        save_key: (string) key of the h5py file for the latents.
    """
    model, _ = set_model(opt, cuda_use=opt.cuda_use)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['model'])
    modis_data = np.repeat(modis_data, 3, axis=1)
    num_samples = modis_data.shape[0]
    batch_size = opt.batch_size
    num_steps = num_samples // batch_size
    remainder = num_samples % batch_size
    latents_df = pd.DataFrame()
    with torch.no_grad():
        for i in trange(num_steps):
            image_batch = modis_data[i*batch_size: (i+1)*batch_size]
            image_tensor = torch.tensor(image_batch)
            if opt.cuda_use and torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            latents_tensor = model(image_tensor)
            latents_numpy = latents_tensor.cpu().numpy()
            latents_df = pd.concat([latents_df, pd.DataFrame(latents_numpy)], ignore_index=True)
        if remainder:
            image_remainder = modis_data[-remainder:]
            image_tensor = torch.tensor(image_remainder)
            if opt.cuda_use and torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            latents_tensor = model(image_tensor)
            latents_numpy = latents_tensor.cpu().numpy()
            latents_df = pd.concat([latents_df, pd.DataFrame(latents_numpy)], ignore_index=True)
            latents_numpy = latents_df.values
    with h5py.File(save_path, 'a') as file:
        file.create_dataset(save_key, data=latents_numpy)
        
def main_evaluate(opt_path):
    """
    This function is used to obtain the latents of the trained models.
    Args:
        opt_path: (str) option file path.
        model_path: (str)
        model_list: (list)
        save_key: (str)
        save_path: (str)
        save_base: (str) base name for the saving file
    """
    opt = option_preprocess(Params(opt_path))
    
    # get the model files in the model directory.
    model_files = os.listdir(opt.save_folder)
    model_name_list = [f.split(".")[0] for f in model_files if f.endswith(".pth")]

    data_file = os.path.join(opt.data_folder, os.listdir(opt.data_folder)[0])
    with h5py.File(data_file, 'r') as file:
        dataset_train = file['train'][:]
        dataset_valid = file['valid'][:]
    print("Reading data is done.")
    
    if not os.path.isdir(opt.latents_folder):
        os.makedirs(opt.latents_folder)
    
    key_train, key_valid = "train", "valid"
    
    for i, model_name in enumerate(model_name_list):
        model_path = os.path.join(opt.save_folder, model_files[i])
        file_name = "_".join([model_name, "latents.h5"])
        latents_path = os.path.join(opt.latents_folder, file_name) 
        model_latents_extract(opt, dataset_train, model_path, latents_path, key_train)
        print("Extraction of Latents of train set is done.")
        model_latents_extract(opt, dataset_valid, model_path, latents_path, key_valid)
        print("Extraction of Latents of valid set is done.")
        
if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    
    # run the 'main_train()' function.
    if args.func_flag == 'train':
        print("Training Starts.")
        main_train(args.opt_path)
        print("Training Ends.")
    
    # run the "main_evaluate()" function.
    if args.func_flag == 'evaluate':
        print("Evaluation Starts.")
        main_evaluate(args.opt_path)
        print("Evaluation Ends.")
