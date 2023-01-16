""" Figures for SSL paper on MODIS """
from dataclasses import replace
from datetime import datetime
import os, sys
import numpy as np
import scipy
from scipy import stats
from urllib.parse import urlparse
import datetime

import argparse

import healpy as hp

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Ellipse


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import h5py

from ulmo import plotting
from ulmo.utils import utils as utils

from ulmo import io as ulmo_io
from ulmo.ssl import single_image as ssl_simage
from ulmo.ssl import ssl_umap
from ulmo.utils import image_utils

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import ssl_paper_analy

def fig_uniform_gallery(table:str, local:bool=False,
                     umap_dim=2,
                     umap_comp='S0,S1',
                     seed=1234,
                     min_pts=1000,
                     nxy=3):

    if seed is not None:
        np.random.seed(seed)

    # Load
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, table=table)

    # UMAP
    umap_keys = ssl_paper_analy.gen_umap_keys(
        umap_dim, umap_comp)

    # Outfile
    if 'all' in table:
        outfile = 'fig_uniform_gallery_DTall.png'


    # Grab the cutouts
    cutouts = ssl_umap.cutouts_on_umap_grid(
        modis_tbl, nxy, umap_keys, min_pts=min_pts)
    ncutouts = len(cutouts)

    embed(header='83 of figs_talk.py')

#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    if flg_fig & (2 ** 0):
        fig_uniform_gallery('96clear_v4_DTall', local=True)


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        flg_fig += 2 ** 0  # Gallery of 9 with DT = all
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)