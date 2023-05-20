import os
import pandas
import numpy as np

from IPython import embed

def create_table(models = [10, 35, 50, 75],
    masks = [10, 20, 30, 40, 50],
    data_filepath=os.path.join(
        os.getenv('OS_OGCM'), 
        'LLC', 'Enki', 'Tables', 'MAE_LLC_valid_nonoise.parquet'),
    nbatch:int=20):
    # load tables
    table = pandas.read_parquet(data_filepath, engine='pyarrow')
    table = table[table['LL'].notna()]
    table = table.sort_values(by=['LL'])
    

    # calculate median LL (need to fix this)
    x = ["" for i in range(nbatch)]
    istep = len(table.index)//nbatch
    for i in range(nbatch):
        start = i*istep
        end = (i+1)*istep
        avg = int((start + end)/2)
        x[i] = table.iloc[avg]['LL']
    
    avgs = pandas.DataFrame(x, columns=['median_LL'])

    # calculate batch averages
    for t in models:
        for p in masks:
            index = 'rms_t{}_p{}'.format(t, p)
            avg_rms = calc_batch_RMSE(table, t, p, 100/nbatch)
            avgs[index] = avg_rms

    return avgs
        
def calc_batch_RMSE(table, t, p, batch_percent:float = 10.):
    """
    Calculates RMSE in batches. Handles extra by adding them to final batch
    so pick reasonable batch sizes that won't leave a lot of extra 
    table:   LL table (sorted)
    t:       mask ratio during training
    p:       mask ratio during reconstruction
    """
    num_imgs = len(table.index)
    batch_size = int(num_imgs*batch_percent/100) # size of batch
    num_batches = num_imgs // batch_size # batches to run excluding final batch
    final_batch = num_imgs-batch_size*(num_batches-1)
    label = 'RMS_t{t}_p{p}'.format(t=t, p=p)

    print('Calculating batches for t{t}_p{p}'.format(t=t, p=p))
    RMSE = np.empty(num_batches, dtype=np.float64)
    for batch in range(num_batches-1):
        start = batch*batch_size
        end = start + batch_size-1
        arr = table[label].to_numpy()
        RMSE[batch] = sum(arr[start:end])/batch_size
        #RMSE[batch] = calculate_RMSE(table, label, start, end, batch_size)
    
    RMSE[num_batches-1] = RMSE[batch] = sum(arr[batch_size*(num_batches-1):num_imgs-1])/final_batch
    return RMSE