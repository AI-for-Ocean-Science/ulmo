import os
import numpy as np
import pandas

from ulmo import io as ulmo_io

from IPython import embed

def path_to_tables(dataset:str):
    """ Generate path to tables

    Args:
        dataset (str): 

    Returns:
        str: the path
    """

    if dataset == 'modis': 
        path = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Tables')


    return path

    

def load(tbl_file:str,
         local=False, cuts:str=None, 
         region:str=None, percentiles:list=None):
    """Load up a data table and (usually) cut it down

    Args:
        tbl_file (str): Table filename
        local (bool, optional): Load file on local harddrive?. Defaults to False.
        cuts (str, optional): Named cuts. Defaults to None.
            inliers: Restrict to LL = [200,400]
        region (str, optional): Cut on geographic region. Defaults to None.
            Brazil, GS, Med
        percentiles (list, optional): Cut on percentiles of LL. Defaults to None.

    Raises:
        IOError: _description_

    Returns:
        pandas.Dataframe: table
    """
    # Load
    tbl = ulmo_io.load_main_table(tbl_file)

    # DT
    if 'DT' not in tbl.keys():
        tbl['DT'] = tbl.T90 - tbl.T10
    tbl['logDT'] = np.log10(tbl.DT)
    tbl['lowDT'] = tbl.mean_temperature - tbl.T10
    tbl['absDT'] = np.abs(tbl.T90) - np.abs(tbl.T10)

    # Slopes
    if 'zonal_slope' in tbl.keys():
        tbl['min_slope'] = np.minimum(
            tbl.zonal_slope, tbl.merid_slope)

    # Cut
    if 'LL' in tbl.keys():
        goodLL = np.isfinite(tbl.LL)
    else:
        goodLL = np.ones(len(tbl), dtype=bool)
    if cuts is None:
        good = goodLL
    elif cuts == 'inliers':
        inliers = (tbl.LL > 200.) & (tbl.LL < 400)
        good = goodLL & inliers
    tbl = tbl[good].copy()

    # Region?
    if region is None:
        pass
    elif region == 'brazil':
        # Brazil
        in_brazil = ((np.abs(tbl.lon.values + 57.5) < 10.)  & 
            (np.abs(tbl.lat.values + 43.0) < 10))
        in_DT = np.abs(tbl.DT - 2.05) < 0.05
        tbl = tbl[in_brazil & in_DT].copy()
    elif region == 'GS':
        # Gulf Stream
        in_GS = ((np.abs(tbl.lon.values + 69.) < 3.)  & 
            (np.abs(tbl.lat.values - 39.0) < 1))
        tbl = tbl[in_GS].copy()
    elif region == 'Med':
        # Mediterranean
        in_Med = ((tbl.lon > -5.) & (tbl.lon < 30.) &
            (np.abs(tbl.lat.values - 36.0) < 5))
        tbl = tbl[in_Med].copy()
    else: 
        raise IOError(f"Bad region! {region}")

    # Percentiles
    if percentiles is not None:
        LL_p = np.percentile(tbl.LL, percentiles)
        cut_p = (tbl.LL < LL_p[0]) | (tbl.LL > LL_p[1])
        tbl = tbl[cut_p].copy()

    return tbl

def parse_metric(metric:str, tbl:pandas.DataFrame):
    """ Evaluate a given metric
    And provide a label

    Args:
        metric (str): Metric to evaluate
        tbl (pandas.DataFrame): Table of cutouts

    Raises:
        IOError: _description_

    Returns:
        tuple: evaluated metric (np.ndarray), label (str)
    """
    # Metric
    lmetric = metric
    if metric == 'LL':
        values = tbl.LL 
    elif metric == 'logDT':
        values = np.log10(tbl.DT.values)
        lmetric = r'$\log_{10} \, \Delta T$'
    elif metric == 'DT':
        values = tbl.DT.values
        lmetric = r'$\Delta T$'
    elif metric == 'DT40':
        values = tbl.DT40.values
        lmetric = r'$\Delta T_{40}$ (K)'
        #lmetric = r'$\Delta T_{\rm 40}$'
    elif metric == 'stdDT':
        values = tbl.DT.values
        lmetric = r'$\sigma(\Delta T) (K)$'
    elif metric == 'stdDT40':
        values = tbl.DT40.values
        #lmetric = r'$\sigma(\Delta T_{\rm 40}) (K)$'
        lmetric = r'$\sigma(\Delta T_{40}) (K)$'
    elif metric == 'logDT40':
        values = np.log10(tbl.DT40.values)
        lmetric = r'$\log \Delta T_{\rm 40}$'
    elif metric == 'clouds':
        values = tbl.clear_fraction
        lmetric = 'Cloud Coverage'
    elif metric == 'slope':
        lmetric = r'slope ($\alpha_{\rm min}$)'
        values = tbl.min_slope.values
    elif metric == 'meanT':
        lmetric = r'$<T>$ (K)'
        values = tbl.mean_temperature.values
    elif metric == 'abslat':
        lmetric = r'$|$ latitude $|$ (deg)'
        values = np.abs(tbl.lat.values)
    elif metric == 'lon':
        lmetric = r'longitude (deg)'
        values = tbl.lon.values
    elif metric == 'counts':
        lmetric = 'Counts'
        values = np.ones(len(tbl))
    elif metric == 'log10counts':
        lmetric = r'$\log_{10}$ Counts'
        values = np.ones(len(tbl))
    # Frontogenesis
    elif metric == 'FS_Npos':
        lmetric = r'FS_Npos',
        values = tbl[metric].values.astype(int)
    # Scattering
    elif metric == 'S1_iso_4':
        lmetric = r'$S_{1,iso}^4$',
        values = tbl[metric].values
    elif metric == 's21':
        lmetric = r'$s_{21}$'
        values = tbl[metric].values
    elif metric == 's22':
        lmetric = r'$s_{22}$'
        values = tbl[metric].values
    else:
        raise IOError("Bad metric!")

    return lmetric, values