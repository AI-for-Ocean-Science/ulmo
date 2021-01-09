""" Script to grab LLC model data """

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Grab LLC Model data')
    parser.add_argument("tstep", type=int, help="Time step in hours")
    #parser.add_argument("-s","--show", default=False, action="store_true", help="Show pre-processed image?")
    parser.add_argument("--model", type=str, default='LLC4320',
                        help="LLC Model name.  Allowed options are [LLC4320]")
    parser.add_argument("--var", type=str, default='Theta',
                        help="LLC data variable name.  Allowed options are [Theta, U, V, Salt]")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np
    import os
    import warnings

    import xmitgcm.llcreader as llcreader
    from ulmo.llc.slurp import write_xr

    # Load model
    if pargs.model == 'LLC4320':
        model = llcreader.ECCOPortalLLC4320Model()
        tstep_hr = 144  # Time steps per hour

    # Get dataset
    iter_step = tstep_hr*pargs.tstep
    ds = model.get_dataset(varnames=pargs.var.split(','),
                               k_levels=[0], type='latlon',
                               iter_step=iter_step)  
    print("Model is ready")

    # Loop me
    for tt in range(ds.time.size):
        print("Time step = {} of {}".format(tt, ds.time.size))
        #SST = ds_sst.Theta.isel(time=tt, k=0)  # , i=slice(1000,2000), j=slice(1000,2000))
        ds_0 = ds.isel(time=tt, k=0)  # , i=slice(1000,2000), j=slice(1000,2000))
        # Generate outfile name
        outfile = '{:s}_{:s}.nc'.format(pargs.model, 
            str(ds_0.time.values)[:19].replace(':','_'))
        # No clobber
        if os.path.isfile(outfile):
            print("Not clobbering: {}".format(outfile))
            continue
        # Write
        write_xr(ds_0, outfile)
        print("Wrote: {}".format(outfile))


