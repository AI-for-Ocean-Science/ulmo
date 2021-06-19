""" Module for Ulmo analysis on VIIRS 2013"""
import os
from h5py._hl import base
import numpy as np
from pkg_resources import resource_filename

from matplotlib import pyplot as plt

import h5py

from ulmo import io as ulmo_io
from ulmo.plotting import plotting
from ulmo.utils import catalog as cat_utils
from ulmo.ssl import analysis as ssl_analysis

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

    # Stack em
    latents = np.concatenate([latents_train, latents_valid])
    ssl_analysis.do_umap(
        latents, np.arange(latents_train.shape[0]), 
        latents_train.shape[0]+np.arange(latents_valid.shape[0]),
        valid_tbl, fig_root='MODIS_2010_v2', debug=False,
        write_to_file='s3://modis-l2/Tables/MODIS_2010_valid_SSLv2.parquet')


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