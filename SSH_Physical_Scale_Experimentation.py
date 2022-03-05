import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import time

# Variables for file paths
### Example: fn = r"C:\Users\Ben\Desktop\Prochaska Group\SSH Data\SSH_Data_Files\ssh_grids_v1812_1992100212.nc"
fn = "YOUR FILE PATH HERE"

'''
The function takes in a file path for data, coordinates, and a size for a pixel grid and returns a countoured Plate Caree map of a
region specified by the coordinates and plot size.

Code doesn't work well at very large scales, at high latitiudes, and at low longitudes
'''


def SSH_Map_Section(filepath,lat,lon,pixels):

    sqrsizeKm = int(pixels * 18.5) # The data is on a 1/6° grid which is about 18.5 km
    LatLonDeg = (sqrsizeKm / 111) / 2 # This is what will be used to set th eextent of the plot. The "/ 2" is because it adds/subtracts the "radius" from the center point

    # Ceating the bounds to slice the data and set the extent of the plot    
    # Converts °W to °E. Cartopy can use either but the data only takes °E
    if lon < 0:
        lony = lon+360
    else: lony = lon
    
    minlon = abs(int(lony - LatLonDeg))
    maxlon = abs(int(lony + LatLonDeg))
    
    minlat = int(lat - LatLonDeg)
    maxlat = int(lat + LatLonDeg)
    
    # Opening data and slicing spatialy to remove data outside desired bounds
    ds = xr.open_dataset(filepath).sel(Longitude=slice(minlon, maxlon),Latitude=slice(minlat, maxlat))
  
    # Averages time to make the data 2D. Also transposed the axis because for some reason in the raw data they're flipped
    ds = ds['SLA'].mean(dim="Time").transpose()


    # Initializing the plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
   
    # Formatting the title to include information about the plot
    coordinate = str("{0}°N,{1}°E").format(lat,lon) # for formatting the center point to be put into the plot title
    squaresize = str("{0}x{1}Km").format(sqrsizeKm,sqrsizeKm) # for formatting the plot size to be put into the plot title
    ax.set_title("{0} pixel grid ({1}) SSH plot centered at {2}".format(pixels,squaresize,coordinate), size=17)
    
    # Setting up the plot
    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.right_labels = False
    ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree()) # left lon bound, right lon bound, top lat bound, bottom lat bound

   # Making the Color Bar
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    CF = ax.pcolormesh(ds.Longitude, ds.Latitude, ds, transform=ccrs.PlateCarree()) # need this one for contour color
    cbar = plt.colorbar(CF,cax=cax)
    cbar.set_label('SSH (m)', rotation=270)


TimerStart = time.perf_counter()

### Example SSH_Map_Section(fn,40,-50,128)
SSH_Map_Section(fn,,,)    

TimerStop = time.perf_counter() 
RunTime = TimerStop - TimerStart
print(("Run time: {:.2f} seconds").format(RunTime))



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
