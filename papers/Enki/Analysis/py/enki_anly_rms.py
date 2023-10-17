import os
import pandas
import numpy as np

import scipy

from ulmo.enki import enki_utils

from IPython import embed

def calc_median_LL(table:pandas.DataFrame, nbatch:int=20,
                   sort:bool=True):
    # Clean up
    if sort:
        tbl = table[table['LL'].notna()].copy()
        tbl = tbl.sort_values(by=['LL'])
    else:
        tbl = table.copy()

    # calculate median LL (need to fix this)
    x = ["" for i in range(nbatch)]
    istep = len(tbl.index)//nbatch
    starts, ends = [], []
    for i in range(nbatch):
        start = i*istep
        end = (i+1)*istep
        avg = int((start + end)/2)
        x[i] = tbl.iloc[avg]['LL']
        # Save
        starts.append(tbl.iloc[start]['LL'])
        ends.append(tbl.iloc[end]['LL'])

    # Table me
    medL = pandas.DataFrame(x, columns=['median_LL'])
    # Return
    return medL, starts, ends

def create_llc_table(
    models = [10, 35, 50, 75],
    masks = [10, 20, 30, 40, 50],
    table:pandas.DataFrame=None,
    method:str=None,
    data_filepath=os.path.join(
        os.getenv('OS_OGCM'), 
        'LLC', 'Enki', 'Tables', 'MAE_LLC_valid_nonoise.parquet'),
    nbatch:int=20):

    # load tables
    if table is None:
        table = pandas.read_parquet(data_filepath, engine='pyarrow')

    table = table[table['LL'].notna()].copy()
    table = table.sort_values(by=['LL'])

    # Median LL
    avgs, _, _ = calc_median_LL(table, nbatch=nbatch, sort=False)
    
    # calculate batch averages
    for t in models:
        for p in masks:
            index = 'rms_t{}_p{}'.format(t, p)
            avg_rms = calc_batch_RMSE(table, t, p, 100/nbatch, sort=False,
                                      method=method)
            avgs[index] = avg_rms

    return avgs
        
def calc_batch_RMSE(table, t, p, batch_percent:float = 10.,
                    sort:bool=True, inpaint:bool=False,
                    method:str=None):
    """
    Calculates RMSE in batches sorted by LL. 
    Handles extra by adding them to final batch
    so pick reasonable batch sizes that won't leave a lot of extra 

    table:   LL table 
    t:       mask ratio during training
    p:       mask ratio during reconstruction
    inpaint: if True, use inpainted RMSE
    method:  if not None, use this method to calculate RMSE
    """
    if sort:
        tbl = table[table['LL'].notna()].copy()
        tbl = tbl.sort_values(by=['LL'])
    else:
        tbl = table.copy()

    num_imgs = len(tbl.index)
    batch_size = int(num_imgs*batch_percent/100) # size of batch
    num_batches = num_imgs // batch_size # batches to run excluding final batch
    final_batch = num_imgs-batch_size*(num_batches-1)

    if inpaint:
        key = 'RMS_inpaint_t{t}_p{p}'.format(t=t, p=p)
    elif method is not None:
        key = f'RMS_{method}_t{t}_p{p}'
    else:
        key = 'RMS_t{t}_p{p}'.format(t=t, p=p)

    print('Calculating batches for t{t}_p{p}'.format(t=t, p=p))
    RMSE = np.empty(num_batches, dtype=np.float64)
    for batch in range(num_batches-1):
        start = batch*batch_size
        end = start + batch_size-1
        arr = tbl[key].to_numpy()
        # Deal with NaNs
        good = np.isfinite(arr[start:end])
        RMSE[batch] = np.sum(arr[start:end][good])/np.sum(good)
    
    RMSE[num_batches-1] = RMSE[batch] = sum(arr[batch_size*(num_batches-1):num_imgs-1])/final_batch
    return RMSE

def anly_patches(patch_file:str, nbins:int=32, model:str='std'):
    """ Analyze the patches

    Args:
        patch_file (str): _description_
        nbins (int, optional): _description_. Defaults to 32.
        nfit (int, optional): Number of parameters for the fit. Defaults to 2.

    Returns:
        _type_: _description_
    """

    # Load
    patch_file = os.path.join(os.getenv("OS_OGCM"),
        'LLC', 'Enki', 'Recon', patch_file)

    print(f'Loading: {patch_file}')
    f = np.load(patch_file)
    data = f['data']
    data = data.reshape((data.shape[0]*data.shape[1], 
                         data.shape[2]))

    items = f['items']
    tbl = pandas.DataFrame(data, columns=items)

    metric = 'log10_std_diff'
    stat = 'median'

    x_metric = 'log10_stdT'
    xvalues, x_lbl = enki_utils.parse_metric(tbl, x_metric)

    values, lbl = enki_utils.parse_metric(tbl, metric)

    good1 = np.isfinite(xvalues.values)
    good2 = np.isfinite(values.values)
    good = good1 & good2

    # Do it
    eval_stats, x_edge, ibins = scipy.stats.binned_statistic(
        xvalues.values[good], values.values[good], statistic=stat, bins=nbins)

    # Fit
    x = (x_edge[:-1]+x_edge[1:])/2
    gd_eval = np.isfinite(eval_stats)

    gd_x = 10**x[gd_eval]
    gd_y = 10**eval_stats[gd_eval]

    if model == 'std':
        fit_model = two_param_model 
        p0=[0.01, 10.]
    elif model == 'denom':
        fit_model = denom_model
        p0 = None
    else:
        raise ValueError(f'Unknown model: {model}')
    popt, pcov = scipy.optimize.curve_fit(
        fit_model, gd_x, gd_y, p0=p0, sigma=0.1*gd_y)
    return x_edge, eval_stats, stat, x_lbl, lbl, popt, tbl


def two_param_model(sigT, floor:float, scale:float):
    rmse = (sigT+floor)/scale
    return rmse

def denom_model(sigT, floor:float, scale:float):
    rmse = (sigT+floor)/(np.sqrt(sigT) + scale)
    return rmse
    
# Command line execution
if __name__ == '__main__':
    anly_patches('mae_patches_t10_p20.npz')