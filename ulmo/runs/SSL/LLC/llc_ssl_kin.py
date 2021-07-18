from __future__ import print_function

import time
import os
import h5py
import numpy as np
import pandas as pd
import tqdm
from tqdm.auto import trange
import argparse

import torch

from ulmo import io as ulmo_io
from ulmo.llc import extract 

from ulmo.ssl.util import adjust_learning_rate
from ulmo.ssl.util import set_optimizer, save_model

from ulmo.ssl.train_util import Params, option_preprocess
from ulmo.ssl.train_util import modis_loader_v2, set_model
from ulmo.ssl.train_util import train_model
from ulmo.ssl.latents_extraction import model_latents_extract, build_loader
from ulmo.ssl.latents_extraction import calc_latent

from ulmo.utils import catalog as cat_utils

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
                        help="flag of the function to be execute: 'extract_curl'.")
    args = parser.parse_args()
    
    return args

def main_train(opt_path: str):
    # loading parameters json file
    opt = Params(opt_path)
    opt = option_preprocess(opt)

    # build data loader
    train_loader = modis_loader_v2(opt)

    # build model and criterion
    model, criterion = set_model(opt, cuda_use=opt.cuda_use)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    # read 'user' and 'pin' for comet log
    #with open('/etc/comet-pin-volume/username', 'r') as f:
    #    user = f.read()
    
    #with open('/etc/comet-pin-volume/password', 'r') as f:
    #    pin = f.read()
        
    # comet log
    #experiment = Experiment(
    #        api_key=pin,
    #        project_name="LLC_modis2012_curl", 
    #        workspace=user,
    #)
    #experiment.log_parameters(opt.dict)
    
    # training routine
    for epoch in trange(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train_model(train_loader, model, criterion, optimizer, epoch, opt, cuda_use=opt.cuda_use)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        # comet
        #experiment.log_metric('loss', loss, step=epoch)
        #experiment.log_metric('learning_rate', optimizer.param_groups[0]['lr'], step=epoch)
        
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    

    
def model_latents_extract(opt, modis_data_file, modis_partition, 
                          model_path, save_path, 
                          save_key,
                          remove_module=True):
    """
    This function is used to obtain the latents of the training data.
    Args:
        opt: (Parameters) parameters used to create the model.
        modis_data_file: (str) path of modis_data_file.
        modis_partition: (str) key of the h5py file.
        model_path: (string) path of the saved model file.
        save_path: (string) path for saving the latents.
        save_key: (string) path for the key of the saved latents.
    """
    using_gpu = torch.cuda.is_available()
    model, _ = set_model(opt, cuda_use=using_gpu)
    if not using_gpu:
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model_dict = torch.load(model_path)

    if remove_module:
        new_dict = {}
        for key in model_dict['model'].keys():
            new_dict[key.replace('module.','')] = model_dict['model'][key]
        model.load_state_dict(new_dict)
    else:
        model.load_state_dict(model_dict['model'])
    print("Model loaded")

    # Data
    _, loader = build_loader(modis_data_file, modis_partition)

    print("Beginning to evaluate")
    model.eval()
    with torch.no_grad():
        latents_numpy = [calc_latent(
                model, data[0], using_gpu) for data in tqdm.tqdm(
                loader, total=len(loader), unit='batch', 
                desc='Computing latents.')]
        
    with h5py.File(save_path, 'w') as file:
        file.create_dataset(save_key, data=np.concatenate(latents_numpy))
    print("Wrote: {}".format(save_path))
    
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
    print('opt file is read.')
    # get the model files in the model directory.
    model_files = os.listdir(opt.save_folder)
    model_name_list = [f.split(".")[0] for f in model_files if f.endswith(".pth")]

    data_file = os.path.join(opt.data_folder, os.listdir(opt.data_folder)[0])
    
    if not os.path.isdir(opt.latents_folder):
        os.makedirs(opt.latents_folder)
    print('folder for latents is created.')
    
    key_train, key_valid = "train", "valid"
    
    for i, model_name in enumerate(model_name_list):
        model_path = os.path.join(opt.save_folder, model_files[i])
        file_name = "_".join([model_name, "latents.h5"])
        latents_path = os.path.join(opt.latents_folder, file_name)
        data_path = os.path.join(opt.data_folder, os.listdir(opt.data_folder)[0])
        if opt.eval_key == 'train':
            print("Extraction of latents of train set starts.")
            model_latents_extract(opt, data_path, 'train', model_path, latents_path, 'train')
            print("Extraction of latents of train set is done.")
        elif opt.eval_key == 'valid':
            print("Extraction of latents of valid set starts.")
            model_latents_extract(opt, data_path, 'valid', model_path, latents_path, 'valid')
            print("Extraction of latents of valid set is done.")
        elif opt.eval_key == 'train_valid':
            print("Extraction of latents of train set starts.")
            model_latents_extract(opt, data_path, 'train', model_path, latents_path, 'train')
            print("Extraction of latents of train set is done.")
            print("Extraction of latents of valid set is started.")
            model_latents_extract(opt, data_path, 'eval', model_path, latents_path, 'eval')
            print("Extraction of latents of valid set is done.")
        else:
            raise Exception("opt.eval_dataset is not right!")
    
def extract_curl(debug=True):

    orig_tbl_file = 's3://llc/Tables/test_noise_modis2012.parquet'
    tbl_file = 's3://llc/Tables/modis2012_kin_curl.parquet'
    root_file = 'LLC_modis2012_curl_preproc.h5'
    llc_table = ulmo_io.load_main_table(orig_tbl_file)

    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://llc/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    # Run it
    llc_table = extract.velocity_field(llc_table, 'curl',
                                pp_local_file, 
                                'llc_std',
                                s3_file=pp_s3_file,
                                debug=debug)
    # Vet
    assert cat_utils.vet_main_table(llc_table, cut_prefix='modis_')

    # Final write
    if not debug:
        ulmo_io.write_main_table(llc_table, tbl_file)
    


if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    
    # run the 'main_train()' function.
    if args.func_flag == 'extract_curl':
        print("Extraction starts.")
        extract_curl(debug=False)
        print("Extraction Ends.")
        
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