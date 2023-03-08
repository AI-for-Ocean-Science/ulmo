""" Run SSL Analysis specific to the paper """

import os
import numpy as np

import pandas

from matplotlib import pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import statsmodels.formula.api as smf

from ulmo import io as ulmo_io
from ulmo.ssl import defs as ulmo_ssl_defs

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

# Geography
geo_regions = ulmo_ssl_defs.geo_regions

# UMAP ranges for the paper
umap_rngs_dict = {}
#umap_rngs_dict['weak_DT15'] = [[1.5,3.],  # DT15, old UMAP
#                          [1.5,3]]
umap_rngs_dict['weak_DT1'] = [[0, 2.0],  # DT1, new UMAP
                          [-2.5,-0.3]]
#umap_rngs_dict['weak_DT1'] = [[-1, 1.],  # DT1, new UMAP
#                          [-3.,-0.5]]
umap_rngs_dict['strong_DT1'] = [[4.0,8-0.7],  # DT1, new UMAP
                          [2.4,4]]

def lon_to_lbl(lon):
    if lon < 0:
        return '{:d}W'.format(int(-lon))
    else:
        return '{:d}E'.format(int(lon))
        
def lat_to_lbl(lat):
    if lat < 0:
        return '{:d}S'.format(int(-lat))
    else:
        return '{:d}N'.format(int(lat))

def load_modis_tbl(table:str=None, 
                   local=False, cuts:str=None, 
                   region:str=None, percentiles:list=None):
    """Load up the MODIS table and (usually) cut it down

    Args:
        table (str, optional): Code for the table name. Defaults to None.
            std, CF, CF_DT0, ...
        local (bool, optional): Load file on local harddrive?. Defaults to False.
        cuts (str, optional): Named cuts. Defaults to None.
            inliers: Restrict to LL = [200,400]
        region (str, optional): Cut on geographic region. Defaults to None.
            Brazil, GS, Med
        percentiles (list, optional): Cut on percentiles of LL. Defaults to None.

    Raises:
        IOError: _description_

    Returns:
        pandas.Dataframe: MODIS table
    """
    raise DeprecationWarning("Use ulmo.utils/table.load instead")

    # Which file?
    if table is None:
        table = '96' 
    if table == 'std':  # Original; too many clouds
        basename = 'MODIS_L2_std.parquet'
    else:
        # Base 1
        if 'CF' in table:
            base1 = 'MODIS_SSL_cloud_free'
        elif '96_v4' in table:
            base1 = 'MODIS_SSL_v4'
        elif '96' in table:
            base1 = 'MODIS_SSL_96clear'
        # DT
        if 'DT' in table:
            if 'v4' in table:
                base1 = 'MODIS_SSL_96clear_v4'
            dtstr = table.split('_')[-1]
            base2 = '_'+dtstr
        elif 'v4_a' in table:
            base1 = 'MODIS_SSL_96clear_v4'
            dtstr = table.split('_')[-1]
            base2 = '_'+dtstr
        else:
            base2 = ''
        # 
        basename = base1+base2+'.parquet'

    if local:
        tbl_file = os.path.join(os.getenv('SST_OOD'), 'MODIS_L2', 'Tables', basename)
    else:
        tbl_file = 's3://modis-l2/Tables/'+basename

    # Load
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # DT
    if 'DT' not in modis_tbl.keys():
        modis_tbl['DT'] = modis_tbl.T90 - modis_tbl.T10
    modis_tbl['logDT'] = np.log10(modis_tbl.DT)
    modis_tbl['lowDT'] = modis_tbl.mean_temperature - modis_tbl.T10
    modis_tbl['absDT'] = np.abs(modis_tbl.T90) - np.abs(modis_tbl.T10)

    # Slopes
    modis_tbl['min_slope'] = np.minimum(
        modis_tbl.zonal_slope, modis_tbl.merid_slope)

    # Cut
    goodLL = np.isfinite(modis_tbl.LL)
    if cuts is None:
        good = goodLL
    elif cuts == 'inliers':
        inliers = (modis_tbl.LL > 200.) & (modis_tbl.LL < 400)
        good = goodLL & inliers
    modis_tbl = modis_tbl[good].copy()

    # Region?
    if region is None:
        pass
    elif region == 'brazil':
        # Brazil
        in_brazil = ((np.abs(modis_tbl.lon.values + 57.5) < 10.)  & 
            (np.abs(modis_tbl.lat.values + 43.0) < 10))
        in_DT = np.abs(modis_tbl.DT - 2.05) < 0.05
        modis_tbl = modis_tbl[in_brazil & in_DT].copy()
    elif region == 'GS':
        # Gulf Stream
        in_GS = ((np.abs(modis_tbl.lon.values + 69.) < 3.)  & 
            (np.abs(modis_tbl.lat.values - 39.0) < 1))
        modis_tbl = modis_tbl[in_GS].copy()
    elif region == 'Med':
        # Mediterranean
        in_Med = ((modis_tbl.lon > -5.) & (modis_tbl.lon < 30.) &
            (np.abs(modis_tbl.lat.values - 36.0) < 5))
        modis_tbl = modis_tbl[in_Med].copy()
    else: 
        raise IOError(f"Bad region! {region}")

    # Percentiles
    if percentiles is not None:
        LL_p = np.percentile(modis_tbl.LL, percentiles)
        cut_p = (modis_tbl.LL < LL_p[0]) | (modis_tbl.LL > LL_p[1])
        modis_tbl = modis_tbl[cut_p].copy()

    return modis_tbl

def grab_subset(DT:float):
    raise ValueError("Deal with DT40")
    for key, item in ssl_defs.umap_DT.items():
        if item is None:
            raise ValueError("Should not get to all!!")
        if item[1] < 0:
            DTmin = item[0]
            DTmax = 1e9
        else:
            DTmin = item[0] - item[1]
            DTmax = item[0] + item[1]
        #
        if (DT >= DTmin) & (DT < DTmax):
            break

    # Return
    return key

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

def gen_umap_keys(umap_dim:int, umap_comp:str):
    """ Generate the keys for UMAP 

    Args:
        umap_dim (int): dimension of UMAP
        umap_comp (str): 

    Returns:
        umap_keys (tuple): tuple of keys for UMAP
    """
    if umap_dim == 2:
        if 'T1' in umap_comp:
            umap_keys = ('UT1_'+umap_comp[0], 'UT1_'+umap_comp[-1])
        else:
            ps = umap_comp.split(',')
            umap_keys = ('U'+ps[0], 'U'+ps[-1])
    elif umap_dim == 3:
        umap_keys = ('U3_'+umap_comp[0], 'U3_'+umap_comp[-1])
    return umap_keys

