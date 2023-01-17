import numpy as np
#import netCDF4 as nc
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import os
from os.path import exists
import requests
import time

from IPython import embed


## File name builder: format of file name is year mo da hr
## I made lists of str for each component and will loop through to build every possible filename
## Will handle nonexistant file names



# Lists of strings that are parths of the file name
basepath = "https://opendap.jpl.nasa.gov/opendap/SeaSurfaceTopography/merged_alt/L4/cdr_grid/"

ds_beginning = "ssh_grids_v1812_"

year_list = (np.arange(1992,2020,1)).tolist()
ds_year = [str(x) for x in year_list]

ds_month = ['01','02','03','04','05','06','07','08','09','10','11','12']

day_list = (np.arange(1,32,1)).tolist()
ds_day = [str(x) for x in day_list]

ds_ending = "12.nc" 

# Combines all the year, month, and day strings to create potential file names and adds them to a list
time_list = []
for year in ds_year:
    for months in ds_month:
        for days in ds_day:
            time_series = year + months + days
            FullPath = basepath + ds_beginning + time_series + ds_ending
            time_list.append(FullPath)

embed(header='44 of data')

# Removes nonexistant file names
#### WARNING THESE CAN TAKE A LONG TIME!!!!

t0 = time.time() #I want to time the code; this starts the timer

print(len(time_list)) # checks the length of the filename list before removing nonexistant files


'''
# This is one I found online; its fast and removes some file names but Idk if it's doing it correctly or not
for path in time_list:
    if not exists(path):
        time_list.remove(path)
# I ran this and it said there were 10416 files before and 5208 after; that's too many. It took 0.255 sec
'''

'''
# This method tries to open all the potential file neames. If it works it closes the data; if it doesn't, it removes that file name from the list.
for path in time_list:
    try:
        xr.open_dataset(path)
        xr.close_dataset(path)      
    except:
         time_list.remove(path)
# I ran this and it said there were 5208 after that's too many. It took 1435 sec to run (~24 min) It also FILLS the console with text
'''

'''
# This method supposedly check to see if a url is real or not; I think its meant to look for images but maybe it'll work for this
def PathExist(path):
    r = requests.head(path)
    return r.status_code == requests.codes.ok

for path in time_list:
    if not PathExist(path):
        time_list.remove(path)
# I ran this and it said there were 10416 files before and 5817 after; that's too many. It took 1526 sec to run (~25 min)        
'''
        
        
print(len(time_list)) # checks the length of the filename list after removing nonexistant files; should be about 1992        
        
t1 = time.time() # ends the timer
print("Run Time = {:.3f} seconds".format(t1-t0))


#one_data = xr.open_mfdataset(time_list)
#print(one_data)


# This is a function that takes in the data and returns a countoured Plate Caree map with colorbar
def SSH_Map(ds):
    
    print(ds)
    
    try:
        #print(ds)
        #print(ds.variables)
        SSH = ds['SLA'].mean(dim="Time").transpose() # Averages time to make the data 2D. Also transposed the axis because for some reason in the raw data they're flipped
        #print(SSH)
    
        # Setting the x and y bounds for the plot
        lat = ds.variables['Latitude'][:]
        lon = ds.variables['Longitude'][:]
        #print(lat)
    
        # Making the image
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        fig.set_size_inches([8,8])
    
        ax.set_global()
        ##ax.stock_img()
        ax.coastlines(linewidth=0.5)
        ##ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
       # Making the Color Bar
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        CF = ax.contourf(lon, lat, SSH, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(CF,cax=cax)
     
        cbar.set_label('SSH (m)')

    except:
        print("code is wack") # if bad or nonexistant data is offered, this will print instead of failing
        return
        

#SSH_Map(data)