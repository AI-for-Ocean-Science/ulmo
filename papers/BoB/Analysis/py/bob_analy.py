""" Run SSL Analysis specific to the paper """

from pkg_resources import resource_filename
import os, shutil
import numpy as np
import pickle

import h5py
import umap
import pandas

from matplotlib import pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import statsmodels.formula.api as smf

from ulmo import io as ulmo_io
from ulmo.ssl.train_util import option_preprocess
from ulmo.utils import catalog as cat_utils

import ssl_defs

from IPython import embed

if os.getenv('SST_OOD'):
    local_modis_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_L2_std.parquet')
    local_modis_CF_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_SSL_cloud_free.parquet')
    local_modis_CF_DT2_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_SSL_cloud_free_DT2.parquet')
    local_modis_96_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_SSL_96clear.parquet')

# Geo graphical regions
geo_regions = {}
geo_regions['eqpacific'] = dict(
    lons=[-140, -90.],   # W
    lats=[-10, 10.])    # Equitorial 
geo_regions['baybengal'] = dict(
    lons=[79, 95.],   # E
    lats=[16, 23.])    # N
geo_regions['med'] = dict(
    lons=[0, 60.],   # E
    lats=[30, 45.])    # N
geo_regions['global'] = dict(
    lons=[-999., 999.],   # E
    lats=[-999, 999.])    # N
geo_regions['north'] = dict(
    lons=[-999., 999.],   # E
    lats=[0, 999.])    # N




def time_series(df, metric, show=False):
    # Dummy variables for seasonal
    dummy = np.zeros((len(df), 11), dtype=int)
    for i in np.arange(11):
        for j in np.arange(len(df)):
            if df.month.values[j] == i+1:
                dummy[j,i] = 1

    # Setup
    time = np.arange(len(df)) + 1

    # Repack
    data = pandas.DataFrame()
    data['fitme'] = df[metric].values
    data['time'] = time
    dummies = []
    for idum in np.arange(11):
        key = f'dum{idum}'
        dummies.append(key)
        data[key] = dummy[:,idum]

    # Cut Nan
    keep = np.isfinite(df[metric].values)
    data = data[keep].copy()

    # Fit
    formula = "fitme ~ dum0 + dum1 + dum2 + dum3 + dum4 + dum5 + dum6 + dum7 + dum8 + dum9 + dum10 + time"
    glm_model = smf.glm(formula=formula, data=data).fit()#, family=sm.families.Binomial()).fit()

    # Summary
    glm_model.summary()

    # Show?
    if show:
        plt.clf()
        fig = plt.figure(figsize=(12,8))
        #
        ax = plt.gca()
        ax.plot(data['time'], data['values'], 'o', ms=2)
        # Fit
        ax.plot(data['time'], glm_model.fittedvalues)
        #
        plt.show()

    # Build some useful stuff

    # Inter-annual fit
    xval = np.arange(len(df))
    result_dict = {}
    result_dict['slope'] = glm_model.params['time']
    result_dict['slope_err'] = np.sqrt(
        glm_model.cov_params()['time']['time'])


    seas = []
    seas_err = []
    for idum in np.arange(11):
        key = f'dum{idum}'
        # Value
        seas.append(glm_model.params[key])
        # Error
        seas_err.append(np.sqrt(
            glm_model.cov_params()[key][key]))

    # Add em all up
    yval = glm_model.params['Intercept'] + xval * glm_model.params['time'] + (
        np.mean(seas))
    result_dict['trend_yvals'] = yval

    # Add one more
    seas.append(0.)
    seas_err.append(0.)
    result_dict['seasonal'] = seas
    result_dict['seasonal_err'] = seas_err

    # Return
    return glm_model, result_dict