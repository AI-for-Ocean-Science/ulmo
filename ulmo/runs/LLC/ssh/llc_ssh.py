import time
import os
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import trange
import argparse


import torch

import xmitgcm.llcreader as llcreader

from ulmo.llc.slurp import write_xr

from IPython import embed

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("LLC SSH")
    parser.add_argument("--task", type=str,
                        help="task to execute: 'download','evaluate', 'umap'.")
    #parser.add_argument("--year", type=int, help="Year to work on")
    #parser.add_argument("--n_cores", type=int, help="Number of CPU to use")
    #parser.add_argument("--day", type=int, default=1, help="Day to start extraction from")
    args = parser.parse_args()

    return args

def llc_download(pargs, model_name='LLC4320', tstep=6, istart=0,
                 varnames=['Theta','U','V','W','Salt','Eta']): 
    if model_name == 'LLC4320':
        model = llcreader.ECCOPortalLLC4320Model()
        tstep_hr = 144  # Time steps per hour

    # Get dataset
    iter_step = tstep_hr*tstep
    ds = model.get_dataset(
        varnames=varnames, k_levels=[0], type='latlon', 
        iter_step=iter_step)

    tsize = ds.time.size
    print("Model is ready")

    # Loop me
    for tt in range(istart, tsize):
        # Get dataset
        iter_step = tstep_hr*tstep
        ds = model.get_dataset(varnames=varnames,
                                k_levels=[0], type='latlon',
                               iter_step=iter_step)
        #
        print("Time step = {} of {}".format(tt, ds.time.size))
        #SST = ds_sst.Theta.isel(time=tt, k=0)  # , i=slice(1000,2000), j=slice(1000,2000))
        ds_0 = ds.isel(time=tt, k=0)  # , i=slice(1000,2000), j=slice(1000,2000))
        # Generate outfile name
        outfile = '{:s}_{:s}.nc'.format(model_name,
            str(ds_0.time.values)[:19].replace(':','_'))
        # No clobber
        if os.path.isfile(outfile):
            print("Not clobbering: {}".format(outfile))
            continue
        # Write
        write_xr(ds_0, outfile)
        print("Wrote: {}".format(outfile))
        del(ds)
        embed(header='71 of download')


# ulmo_grab_llc 12 --var Theta,U,V,W,Salt --istart 480

if __name__ == "__main__":
    # get the argument of training.
    pargs = parse_option()
    
    # run the 'extract_curl()' function.
    if pargs.task == 'download':
        print("Download starts.")
        llc_download(pargs)
        print("Download Ends.")
    