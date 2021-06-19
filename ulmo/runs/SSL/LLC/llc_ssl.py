from __future__ import print_function

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
from ulmo.ssl.train_util import modis_loader_v2, set_model
from ulmo.ssl.train_util import train_model

from ulmo import io as ulmo_io
from ulmo.ssl import analysis as ssl_analysis

from IPython import embed

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("argument for training.")
    parser.add_argument("--opt_path", type=str, help="path of 'opt.json' file.")
    parser.add_argument("--func_flag", type=str, 
                        help="flag of the function to be execute: 'train','evaluate', 'umap'.")
    args = parser.parse_args()
    
    return args

def main_train(opt_path: str):
    from comet_ml import Experiment
    # loading parameters json file
    opt = Params(opt_path)
    opt = option_preprocess(opt)

    # build data loader
    train_loader = modis_loader_v2(opt)

    # build model and criterion
    model, criterion = set_model(opt, cuda_use=opt.cuda_use)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    # comet
    raise RuntimeError("The next lines were 'fixed'.  You may need to fix them back")
    experiment = Experiment(
            project_name="llc_modis_2012", 
            workspace="edwkuo",
    )
    experiment.log_parameters(opt.dict)
    
    # training routine
    for epoch in trange(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train_model(train_loader, model, criterion, optimizer, epoch, opt, cuda_use=opt.cuda_use)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        # comet
        experiment.log_metric('loss', loss, step=epoch)
        experiment.log_metric('learning_rate', optimizer.param_groups[0]['lr'], step=epoch)
        
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
    #num_samples = 50 
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
    
    if opt.eval_key == 'train':
        with h5py.File(data_file, 'r') as file:
            dataset_train = file['train'][:]
        print("Reading train data is done.")
    elif opt.eval_key == 'valid':
        with h5py.File(data_file, 'r') as file:
            dataset_valid = file['valid'][:]
        print("Reading eval data is done.")
    elif opt.eval_key == 'train_valid':
        with h5py.File(data_file, 'r') as file:
            dataset_train = file['train'][:]
            dataset_valid = file['valid'][:]
        print("Reading data is done.")
    else:
        raise Exception("opt.eval_datset is not right!")
    
    if not os.path.isdir(opt.latents_folder):
        os.makedirs(opt.latents_folder)
    
    key_train, key_valid = "train", "valid"
    
    for i, model_name in enumerate(model_name_list):
        model_path = os.path.join(opt.save_folder, model_files[i])
        file_name = "_".join([model_name, "latents.h5"])
        latents_path = os.path.join(opt.latents_folder, file_name)
        if opt.eval_key == 'train':
            print("Extraction of latents of train set is started.")
            model_latents_extract(opt, dataset_train, model_path, latents_path, key_train)
            print("Extraction of latents of train set is done.")
        elif opt.eval_key == 'valid':
            print("Extraction of latents of valid set is started.")
            model_latents_extract(opt, dataset_valid, model_path, latents_path, key_valid)
            print("Extraction of latents of valid set is done.")
        elif opt.eval_key == 'train_valid':
            print("Extraction of latents of train set is started.")
            model_latents_extract(opt, dataset_train, model_path, latents_path, key_train)
            print("Extraction of latents of train set is done.")
            print("Extraction of latents of valid set is started.")
            model_latents_extract(opt, dataset_valid, model_path, latents_path, key_valid)
            print("Extraction of latents of valid set is done.")
        else:
            raise Exception("opt.eval_datset is not right!")


def generate_umap(debug=False, orig=False):
    # Latents file (subject to move)
    latents_file = 's3://llc/LLC_MODIS_2012_latents/last_latents.h5'

    # Load em in
    basefile = os.path.basename(latents_file)
    if not os.path.isfile(basefile):
        print("Downloading latents (this is *much* faster than s3 access)...")
        ulmo_io.download_file_from_s3(basefile, latents_file)
        print("Done")
    hf = h5py.File(basefile, 'r')

    latents = hf['valid'][:]
    print("Latents loaded")

    # Table (useful for valid only)
    valid_tbl = ulmo_io.load_main_table('s3://llc/Tables/test_noise_modis2012.parquet')

    # Check
    assert latents.shape[0] == len(valid_tbl)

    # Pick 150,000 random for UMAP training
    train = np.random.choice(len(valid_tbl), size=150000)

    # Stack em
    ssl_analysis.do_umap(
        latents, train, np.arange(len(valid_tbl)),
        valid_tbl, fig_root='LLC_v1', debug=False,
        write_to_file='s3://llc/Tables/LLC_MODIS2012_SSL_v1.parquet')

            
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

    # run the "main_evaluate()" function.
    if args.func_flag == 'umap':
        print("Generating the umap")
        generate_umap()