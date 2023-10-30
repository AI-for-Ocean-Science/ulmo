""" F_S Tests """

import os
import numpy as np
import pandas

import xmitgcm
import xmitgcm.llcreader as llcreader

import xarray

from matplotlib import pyplot as plt
import seaborn as sns

from gsw import density

from ulmo import io as ulmo_io 
from ulmo.llc import plotting as llc_plotting
from ulmo.llc import io as llc_io
from ulmo.llc import kinematics
from ulmo.plotting import plotting
from ulmo.llc.slurp import write_xr
from ulmo.utils import image_utils

from ulmo import io as ulmo_io

from IPython import embed

def grab_images(cutout:pandas.Series, npzroot:str, dtimes:list=None, 
                all_var:str='Theta,U,V,W,Salt'):

    if dtimes is None:
        dtimes = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2,
                  -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Init
    model = llcreader.ECCOPortalLLC4320Model()
    tstep_hr = 144
    iter_step = tstep_hr*1
    ds = model.get_dataset(varnames=all_var.split(','),
                           k_levels=[0], type='latlon',
                           iter_step=iter_step)

    i_cutout = np.argmin(np.abs(pandas.to_datetime(ds.time.data) - cutout.datetime))

    # Grab em
    outfiles = []
    print("Grabbing images")
    print("")
    for toff in dtimes:
        ds_0 = ds.isel(time=i_cutout+toff, k=0)
        # Outfile
        outroot = '{:s}_{:s}.nc'.format('LLC4320',
                str(ds_0.time.values)[:19].replace(':','_'))
        
        outfile = os.path.join('/home/xavier/Projects/Oceanography/OGCM/LLC/data/ThetaUVWSalt',
                            outroot)
        outfiles.append(outfile)
        if os.path.isfile(outfile):
            print(f"Skipping: {outfile}")
            continue
        # 
        write_xr(ds_0, outfile)
        print("Wrote: {}".format(outfile))

    # Cutouts
    images = {}
    for dtime, outfile in zip(dtimes, outfiles):
        ds_0 = xarray.open_dataset(outfile)
        # Cutout
        theta = ds_0.Theta.data[cutout.row:cutout.row+64, cutout.col:cutout.col+64]
        U = ds_0.U.data[cutout.row:cutout.row+64, cutout.col:cutout.col+64]
        V = ds_0.V.data[cutout.row:cutout.row+64, cutout.col:cutout.col+64]
        Salt = ds_0.Salt.data[cutout.row:cutout.row+64, cutout.col:cutout.col+64]

        # F_S
        F_s = kinematics.calc_F_s(U,V,theta,Salt)

        # Record
        images[f'theta_{dtime}'] = theta
        images[f'FS_{dtime}'] = F_s

    # Write
    np.savez(npzroot, **images)
    print(f"Wrote: {npzroot}.npz")


if __name__ == '__main__':

    # ###############################
    # First test with Peter

    # Load
    llc_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Nenya', 'Tables',
                        'LLC_A_Nenya_v1_DT15.parquet')
    llc_tbl = ulmo_io.load_main_table(llc_file)

    # Grab the cutout
    Ucen = [4.2, 8.8]
    cutU = (np.abs(llc_tbl.US0-Ucen[0]) < 0.2) & (
        np.abs(llc_tbl.US1-Ucen[1]) < 0.2) & np.isfinite(llc_tbl.FS_Npos)

    sub_tbl = llc_tbl[cutU]
    sub_tbl = sub_tbl.sort_values(by=['FS_Npos'], ascending=False)
    cutout = sub_tbl.iloc[1]

    # Proceed
    npzroot = f'images_for_peter_DT15_{cutout.pp_idx}'
    grab_images(cutout, npzroot)
    
