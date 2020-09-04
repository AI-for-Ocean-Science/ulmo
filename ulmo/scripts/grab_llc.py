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
                        help="LLC data variable name.  Allowed options are [Theta]")

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
    from ulmo.llc.slurp import write_sst

    # Load model
    if pargs.model == 'LLC4320':
        model = llcreader.ECCOPortalLLC4320Model()
        tstep_hr = 144  # Time steps per hour

    # Get dataset
    iter_step = tstep_hr*pargs.tstep
    ds_sst = model.get_dataset(varnames=[pargs.var], k_levels=[0], type='latlon',
                               iter_step=iter_step)  # , iter_step=960)  # Every 12 hours
    print("Model is ready")

    # Loop me
    for tt in range(ds_sst.time.size):
        print("Time step = {} of {}".format(tt, ds_sst.time.size))
        SST = ds_sst.Theta.isel(time=tt, k=0)  # , i=slice(1000,2000), j=slice(1000,2000))
        # Generate outfile name
        outfile = '{:s}_{:s}.nc'.format(pargs.model, str(SST.time.values)[:19])
        # No clobber
        if os.path.isfile(outfile):
            continue
        # Write
        write_sst(SST, outfile)
        print("Wrote: {}".format(outfile))


