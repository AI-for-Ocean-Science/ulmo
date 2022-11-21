""" Script to find a set of similar images based on SSL + UMAP """

# Have to do this here, so odd..
import pickle

from ulmo import io as ulmo_io

import os

from IPython import embed


def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Grab Similar SSL Images')
    parser.add_argument("pp_file", type=str, help="Pre-process file; can be s3")
    parser.add_argument("--pp_idx", default=None, type=int, help="Pre-process identifier")
    parser.add_argument("--opt_path", type=str, help="Name of parameter file for SSL model")
    parser.add_argument("--method", type=str, default='DT_UMAP', help="Approach to image finding")
    parser.add_argument("--model", type=str, default='v4', help="Model")
    #parser.add_argument("dataset", type=str, help="Name of dataset [LLC]")
    parser.add_argument("--img_path", default=None, type=str, help="Path of the target image")
    parser.add_argument("--num_imgs", default=5, type=int, help="Number of images to be searched")
    parser.add_argument("--s3", default=False, action='store_true',
                        help="Over-ride errors (as possible)? Not recommended")
    parser.add_argument("--remove_model", default=False, action='store_true',
                        help="Remove the model?")


    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def build_gallery(pargs, img, data_tbl, srt, local=True, 
                  outfile='similar_images.png', n_new=5):

    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    import numpy as np

    from ulmo.plotting import plotting
    from ulmo.utils import image_utils

    # Grab the new images
    new_imgs = []

    cutout = data_tbl.iloc[srt.values[0]]

    for idx in srt[0:n_new]:
        cutout = data_tbl.iloc[idx]
        if local:
            local_file = os.path.join(os.getenv('SST_OOD'),
                                 'MODIS_L2', 'PreProc',
                                 os.path.basename(cutout.pp_file))
        else:
            local_file = None
        print("Grabbing {} from {}".format(cutout.pp_idx, os.path.basename(cutout.pp_file)))
        iimg = image_utils.grab_image(cutout, close=True,
                                      local_file=local_file)
        new_imgs.append(iimg)
    
    _, cm = plotting.load_palette()
    fig = plt.figure(figsize=(12, 6))
    plt.clf()
    n_col = 3
    n_rows = n_new // n_col + 1 if n_new % n_col else n_new // n_col
    #gs = gridspec.GridSpec(2,3)
    gs = gridspec.GridSpec(n_rows,n_col)
    

    # Input image
    #vmin, vmax = img.min(), img.max()

    def make_one(timg, idx):
        vmin, vmax = timg.min(), timg.max()
        ax = plt.subplot(gs[idx])
        sns.heatmap(np.flipud(timg), ax=ax, cmap=cm,
               vmin=vmin, vmax=vmax, cbar=False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_aspect('equal')
    make_one(img, 0)

    # Rest
    for kk in range(n_new):
        make_one(new_imgs[kk], kk+1)

    # Layout and save
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=400)
    plt.close()
    print('Wrote {:s}'.format(outfile))
    
def main_umap(pargs):
    """ Run
    """
    # Continue
    import numpy as np
    import os
    from pkg_resources import resource_filename

    import torch
    import h5py
    
    from PIL import Image

    from ulmo import io as ulmo_io
    from ulmo.ssl.train_util import option_preprocess
    from ulmo.ssl import analyze_image
    from ulmo.ssl import umap as ssl_umap

    # Prep
    model_file = None
    if pargs.model == 'LLC' or pargs.model == 'LLC_local':
        model_file = 's3://llc/SSL/LLC_MODIS_2012_model/SimCLR_LLC_MODIS_2012_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/last.pth'
        opt_path = os.path.join(resource_filename('ulmo', 'runs'),
                                'SSL', 'LLC', 'experiments', 
                                'llc_modis_2012', 'opts.json')
        table_file = 's3://llc/Tables/LLC_MODIS2012_SSL_v1.parquet'
    elif pargs.model == 'CF': 
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'SSL', 'MODIS', 'v2', 'experiments',
            'modis_model_v2', 'opts_cloud_free.json')
    elif pargs.model == 'v4': 
        opt_path= os.path.join(resource_filename('ulmo', 'runs'),
            'SSL', 'MODIS', 'v4', 'opts_ssl_modis_v4.json')
    else:
        raise IOError("Bad model!!")

    opt = option_preprocess(ulmo_io.Params(opt_path))
    if model_file is None:
        model_file = os.path.join(opt.s3_outdir, 
                                  opt.model_folder, 'last.pth')
    
    if not (pargs.pp_idx or pargs.img_path):
        raise IOError("One argument of 'pp_idx' and 'img_path' must be valued.")
    
    if pargs.pp_idx and pargs.img_path:
        raise IOError("Only one argument of 'pp_idx' and 'img_path' can be used.")
        
    # Load the image
    print("Loading image..")
    if pargs.pp_idx:
        with ulmo_io.open(pargs.pp_file, 'rb') as f:
            pp_hf = h5py.File(f, 'r')
            img = pp_hf['valid'][pargs.pp_idx:pargs.pp_idx+1, ...]
            #img = pp_hf['valid'][pargs.pp_idx:pargs.pp_idx+64, ...]
        img = np.repeat(img, 3, axis=1)
        print(f"Image with index={pargs.pp_idx} loaded from {pargs.pp_file}") 
    else:
        img = Image.open(pargs.img_path)
        img = np.array(img.resize((64, 64)))
        if img.shape[2] != 3:
            img = np.mean(img, axis=-1, keepdims=True)
            img = np.repeat(img, 3, axis=-1)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        print(f"Target Image is loaded and pre-processd!")
        
    # Generate dataset

    # Calculate latents
    latents = analyze_image.get_latents(
        img, model_file, opt)

    # T90
    timg = img[0,0,...]
    srt = np.argsort(timg.flatten())
    i10 = int(0.1*timg.size)
    i90 = int(0.9*timg.size)
    T10 = timg.flatten()[srt[i10]]
    T90 = timg.flatten()[srt[i90]]
    DT = T90 - T10

    # UMAP me
    print("Embedding")
    latents_mapping, new_table_file = ssl_umap.load(
        pargs, DT=DT)
    if new_table_file is not None:
        table_file = new_table_file
    embedding = latents_mapping.transform(latents)

    # Find the closest
    data_tbl = ulmo_io.load_main_table(table_file)

    dist = (embedding[0,0]-data_tbl.U0)**2 + (
        embedding[0,1]-data_tbl.U1)**2
    srt_dist = np.argsort(dist)

    # Gallery time
    build_gallery(pargs, img[0,0,...], data_tbl, srt_dist, 
                  n_new=pargs.num_imgs)
    
    # Leave no trace
    #if pargs.remove_model:
    #    os.remove(model_base)

if __name__ == "__main__":
    args = parser()
    main_umap(args)

# Eddy
# noise = 480574

