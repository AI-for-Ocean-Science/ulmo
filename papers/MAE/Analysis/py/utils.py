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

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas as pd
import seaborn as sns

import h5py

from ulmo import plotting
from ulmo.utils import utils as utils

from ulmo import io as ulmo_io
from ulmo.ssl import single_image as ssl_simage
from ulmo.ssl import defs as ssl_defs
from ulmo.mae import patch_analysis
from ulmo.utils import image_utils

import sys
import os
import requests

import torch
import numpy as np
import h5py

import matplotlib.pyplot as plt
from PIL import Image
from ulmo.plotting import plotting

# Models:
t10_file = 'data/ENKI_t10.pth'
t35_file = 'data/ENKI_t35.pth'
t50_file = 'data/ENKI_t50.pth'
t75_file = 'data/ENKI_t75.pth'

filepath = 'data/MAE_LLC_valid_nonoise_preproc.h5'
# seed = 1313
# make random mask reproducible (comment out to make it change)
group1 = [209248, 524321, 414040, 610138]
group2 = [245215, 72480, 29518, 569580] 
group3 = [313043, 202716, 15385, 478432] 
group4 = [173629, 426310, 599472, 595621]


indexes = [209248, 524321, 414040, 610138, 245215, 72480, 29518, 569580, 
           313043, 202716, 15385, 478432, 173629, 426310, 599472, 595621]
f = h5py.File(filepath, 'r')

models = [model_mae_10, model_mae_35, model_mae_50, model_mae_75]
mask_ratios = [0.10, 0.35, 0.50, 0.75]

orig_imgs = []
recon_imgs = []
masks = []

for (idx, t, p) in zip(group1, models, mask_ratios):
    orig_img = f['valid'][idx][0]
    orig_img.resize((64,64,1))
    recon_img, mask = run_one_image(orig_img, t, p)
    orig_img = orig_img.squeeze()
    orig_imgs.append(orig_img)
    recon_imgs.append(recon_img)
    masks.append(mask)
    
print("Group 1 finished.")

for (idx, t, p) in zip(group2, models, mask_ratios):
    orig_img = f['valid'][idx][0]
    orig_img.resize((64,64,1))
    recon_img, mask = run_one_image(orig_img, t, p)
    orig_img = orig_img.squeeze()
    orig_imgs.append(orig_img)
    recon_imgs.append(recon_img)
    masks.append(mask)

print("Group 2 finished.")

for (idx, t, p) in zip(group3, models, mask_ratios):
    orig_img = f['valid'][idx][0]
    orig_img.resize((64,64,1))
    recon_img, mask = run_one_image(orig_img, t, p)
    orig_img = orig_img.squeeze()
    orig_imgs.append(orig_img)
    recon_imgs.append(recon_img)
    masks.append(mask)

print("Group 3 finished.")
    
for (idx, t, p) in zip(group4, models, mask_ratios):
    orig_img = f['valid'][idx][0]
    orig_img.resize((64,64,1))
    recon_img, mask = run_one_image(orig_img, t, p)
    orig_img = orig_img.squeeze()
    orig_imgs.append(orig_img)
    recon_imgs.append(recon_img)
    masks.append(mask)

print("Group 4 finished.")