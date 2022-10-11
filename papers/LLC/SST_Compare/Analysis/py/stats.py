""" Methods for stats on the outputs """
import numpy as np
import healpy as hp

import pandas

from ulmo import io as ulmo_io

import sst_compare_utils

from IPython import embed

def healpix_stats(dataset:str, outfile:str, local=False, debug:bool=False,
                  nside:int=64, cut:str=None):

    # Load table
    eval_tbl = sst_compare_utils.load_table(dataset, local=local)

    # Heads
    if cut is not None:
        if cut == 'head':
            cutt = (eval_tbl.datetime.dt.year > 2011) & (
                eval_tbl.datetime.dt.year < 2015) 
        elif cut == 'tail':
            cutt = (eval_tbl.datetime.dt.year > 2017) & (
                eval_tbl.datetime.dt.year < 2021) 
        else:
            raise IOError("Bad cut")
        eval_tbl = eval_tbl[cutt].copy()

    # Healpix
    lats = eval_tbl.lat.values
    lons = eval_tbl.lon.values

    # Healpix coords
    theta = (90 - lats) * np.pi / 180.
    phi = lons * np.pi / 180.
    idx_all = hp.pixelfunc.ang2pix(nside, theta, phi)

    # Stats
    npix_hp = hp.nside2npix(nside)
    sigma = np.zeros(npix_hp)
    mean = np.zeros(npix_hp)
    median = np.zeros(npix_hp)
    counts = np.zeros(npix_hp, dtype=int)

    uni_idx = np.unique(idx_all)
    count = 0
    for idx in uni_idx:
        #print(f'{count} of {len(uni_idx)}')
        gd = idx_all == idx
        if np.sum(gd) > 2:
            sigma[idx] = np.std(eval_tbl[gd].LL.values)
            mean[idx] = np.mean(eval_tbl[gd].LL.values)
            median[idx] = np.median(eval_tbl[gd].LL.values)
            counts[idx] = np.sum(gd)
        count += 1
        if debug and count > 100:
            break

    # Output
    df = pandas.DataFrame()

    df['sigma'] = sigma
    df['mean'] = mean
    df['median'] = median
    df['N'] = counts
    df['idx'] = np.arange(npix_hp)

    df.to_csv(outfile, index=False)
    print(f'Wrote: {outfile}')

# Command line execution
if __name__ == '__main__':

    # All
    #healpix_stats('all_viirs.csv', local=True)#, debug=True)

    # Head/tail
    #healpix_stats('head_viirs.csv', local=True, cut='head')
    #healpix_stats('tail_viirs.csv', local=True, cut='tail')

    # LLC
    #healpix_stats('all_llc.csv', local=True, llc=True)

    # MODIS
    healpix_stats('modis_all', 'all_modis.csv')