""" Module for Ulmo analysis on VIIRS 2013"""
import os
import glob
from h5py._hl import base
import numpy as np
import subprocess 
from pkg_resources import resource_filename

from matplotlib import pyplot as plt

import pandas
import h5py
from skimage.restoration import inpaint 

from sklearn.utils import shuffle

from ulmo import io as ulmo_io
from ulmo.plotting import plotting
from ulmo.utils import catalog as cat_utils

import umap

import subprocess

from IPython import embed

def ssl_v2_umap(debug=False, orig=False):
    # Latents file (subject to move)
    latents_file = 's3://modis-l2/SSL_MODIS_R2019_2010_latents_v2/modis_R2019_2010_latents_last_v2.h5'

    # Load em in
    basefile = os.path.basename(latents_file)
    if not os.path.isfile(basefile):
        print("Downloading latents (this is *much* faster than s3 access)...")
        ulmo_io.download_file_from_s3(basefile, latents_file)
        print("Done")
    hf = h5py.File(basefile, 'r')
    latents_train = hf['modis_latents_v2_train'][:]
    latents_valid = hf['modis_latents_v2_valid'][:]
    print("Latents loaded")
    
    # Table (useful for valid only)
    modis_tbl = ulmo_io.load_main_table('s3://modis-l2/Tables/MODIS_L2_std.parquet')

    # Valid
    valid = modis_tbl.pp_type == 0
    y2010 = modis_tbl.pp_file == 's3://modis-l2/PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'
    valid_tbl = modis_tbl[valid & y2010].copy()
    
    # Check
    assert latents_valid.shape[0] == len(valid_tbl)

    # UMAP me
    print("Running UMAP..")
    reducer_umap = umap.UMAP()
    latents_mapping = reducer_umap.fit(latents_train)
    print("Done")

    # Quick figure
    num_samples = latents_train.shape[0]
    point_size = 20.0 / np.sqrt(num_samples)
    dpi = 100
    width, height = 800, 800

    plt.figure(figsize=(width//dpi, height//dpi))
    plt.scatter(latents_mapping.embedding_[:, 0], 
            latents_mapping.embedding_[:, 1], s=point_size)
    plt.savefig('MODIS_2010_v2_train_UMAP.png')

    # Apply to embedding
    valid_embedding = latents_mapping.transform(latents_valid)

    # New plot
    num_samplesv = latents_valid.shape[0]
    point_sizev = 1.0 / np.sqrt(num_samplesv)
    plt.figure(figsize=(width//dpi, height//dpi))
    ax = plt.gca()
    img = ax.scatter(valid_embedding[:, 0], 
                valid_embedding[:, 1], s=point_sizev,
            c=valid_tbl.LL, cmap='jet', vmin=-1000)
    cb = plt.colorbar(img, pad=0.)
    cb.set_label('LL', fontsize=20.)
    #
    ax.set_xlabel(r'$U_0$')
    ax.set_ylabel(r'$U_1$')
    plotting.set_fontsize(ax, 15.)
    #
    plt.savefig('MODIS_2010_v2_train_UMAP.png', dpi=300)

    # Save to Table
    valid_tbl['U0'] = valid_embedding[:, 0]
    valid_tbl['U1'] = valid_embedding[:, 1]

    # Vet
    assert cat_utils.vet_main_table(valid_tbl, cut_prefix='modis_')

    # Final write
    if not debug:
        ulmo_io.write_main_table(valid_tbl, 's3://modis-l2/Tables/MODIS_2010_valid_SSLv2.parquet')
    

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Run UMAP on the v2 latents for MODIS 2010 (train + valid)
    if flg & (2**0):
        ssl_v2_umap()
        #ssl_v2_umap('valid')


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- v2 MODIS 2010 UMAP
    else:
        flg = sys.argv[1]

    main(flg)