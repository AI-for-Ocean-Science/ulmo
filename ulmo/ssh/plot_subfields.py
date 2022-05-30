import os
import numpy as np

from extract import extract_file
from print_metadata import print_metadata
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt



def SSH_Map(fn,field_size):

    fields, mask, meta = extract_file(fn, sub_grid_step=2, field_size=(field_size, field_size))
    

    lons = [float(item[4]) for item in meta]
    lats = [float(item[3]) for item in meta]
    
    print(len(lats))
    
    # Fixing the coordinates so it displays all the points
    for i in range(len(lons)):
        if lons[i] >= 180:
            lons[i] = lons[i]-360
    
    # Making the image
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    fig.set_size_inches([8,8])
    ax.set_title("{0}x{0} pixel grid sub fields points".format(field_size), size=17)
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.plot(lons, lats, '.', markersize = 1)
    
#print_metadata()

fn = "https://opendap.jpl.nasa.gov/opendap/SeaSurfaceTopography/merged_alt/L4/cdr_grid/ssh_grids_v1812_1992100212.nc"

SSH_Map(fn, 32)


