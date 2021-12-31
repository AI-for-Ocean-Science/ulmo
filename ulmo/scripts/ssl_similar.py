""" Script to grab LLC model data """
### used to set the interpreter searching path
import sys
target_path = '/home/jovyan/ulmo/'
sys.path.append(target_path)

# Have to do this here, so odd..
import pickle
from ulmo import io as ulmo_io
import os


def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Grab Similar SSL Images')
    parser.add_argument("pp_file", type=str, help="Pre-process file; can be s3")
    parser.add_argument("--pp_idx", default=None, type=int, help="Pre-process identifier")
    parser.add_argument("opt_path", type=str, help="Name of parameter file for SSL model")
    parser.add_argument("dataset", type=str, help="Name of dataset [LLC]")
    parser.add_argument("--img_path", default=None, type=str, help="Path of the target image")
    parser.add_argument("--num_imgs", default=5, type=int, help="Number of images to be searched")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs

def load_umap(pargs):
    print("Loading UMAP")
    if pargs.model == 'LLC':
        umap_file = 's3://llc/SSL/LLC_MODIS_2012_model/ssl_LLC_v1_umap.pkl'
    elif pargs.model == 'LLC_local':
        umap_file = './ssl_LLC_v1_umap.pkl'
    else:
        raise IOError("bad model")
    umap_base = os.path.basename(umap_file)
    if not os.path.isfile(umap_base):
        ulmo_io.download_file_from_s3(umap_base, umap_file)
    print("UMAP Loaded")
    # Return
    return pickle.load(ulmo_io.open(umap_base, "rb")) 

def build_gallery(pargs, img, data_tbl, srt, 
                  outfile='similar_images.png', n_new=5):

    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    import numpy as np

    from ulmo.plotting import plotting
    from ulmo.utils import image_utils

    # Grab the new images
    pp_hf = None
    new_imgs = []
    for idx in srt[0:n_new]:
        cutout = data_tbl.iloc[idx]
        print("Grabbing {} from {}".format(cutout.pp_idx, cutout.pp_file))
        iimg, pp_hf = image_utils.grab_image(cutout, close=False,
                                            pp_hf=pp_hf)
        new_imgs.append(iimg)
    pp_hf.close()
    
    _, cm = plotting.load_palette()
    fig = plt.figure(figsize=(12, 6))
    plt.clf()
    n_col = 4
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
    
def main(pargs):
    """ Run
    """
    from IPython import embed
    # UMAP first
    latents_mapping = load_umap(pargs)

    # Continue
    import numpy as np
    import os
    from pkg_resources import resource_filename


    import torch
    import h5py
    
    from PIL import Image

    from ulmo import io as ulmo_io
    from ulmo.ssl.train_util import Params, option_preprocess
    from ulmo.ssl import latents_extraction

    # Prep
    if pargs.model == 'LLC' or pargs.model == 'LLC_local':
        model_file = 's3://llc/SSL/LLC_MODIS_2012_model/SimCLR_LLC_MODIS_2012_resnet50_lr_0.05_decay_0.0001_bsz_64_temp_0.07_trial_0_cosine_warm/last.pth'
        opt_file = os.path.join(resource_filename('ulmo', 'runs'),
                                'SSL', 'LLC', 'experiments', 
                                'llc_modis_2012', 'opts.json')
        table_file = 's3://llc/Tables/LLC_MODIS2012_SSL_v1.parquet'
        
    else:
        raise IOError("Bad model!!")
    
    if not (pargs.pp_idx or pargs.img_path):
        raise IOError("One argument of 'pp_idx' and 'img_path' must be valued.")
    
    if pargs.pp_idx and pargs.img_path:
        raise IOError("Only one argument of 'pp_idx' and 'img_path' can be used.")
        
    # Load the image
    print("Loading image..")
    if pargs.pp_idx:
        with ulmo_io.open(pargs.pp_file, 'rb') as f:
            pp_hf = h5py.File(f, 'r')
        #img = pp_hf['valid'][pargs.pp_idx:pargs.pp_idx+1, ...]
        img = pp_hf['valid'][pargs.pp_idx:pargs.pp_idx+64, ...]
        pp_hf.close()
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
        
    # Build the SSL model
    model_base = os.path.basename(model_file)
    if not os.path.isfile(model_base):
        ulmo_io.download_file_from_s3(model_base, model_file)
    opt = option_preprocess(Params(opt_file))

    # DataLoader
    dset = torch.utils.data.TensorDataset(torch.from_numpy(img).float())
    data_loader = torch.utils.data.DataLoader(
        dset, batch_size=1, shuffle=False, collate_fn=None,
        drop_last=False, num_workers=1)

    # Time to run
    latents = latents_extraction.model_latents_extract(opt, pargs.pp_file, 
                'valid', model_base, None, None,
                loader=data_loader)

    # UMAP me
    print("Embedding")
    embedding = latents_mapping.transform(latents)

    # Find the closest
    data_tbl = ulmo_io.load_main_table(table_file)

    dist = (embedding[0,0]-data_tbl.U0)**2 + (embedding[0,1]-data_tbl.U1)**2
    srt_dist = np.argsort(dist)

    # Gallery time
    build_gallery(pargs, img[0,0,...], data_tbl, srt_dist, n_new=pargs.num_imgs)
    
    # Leave no trace
    os.remove(model_base)

if __name__ == "__main__":
    args = parser()
    main(args)

# Eddy
# noise = 480574

