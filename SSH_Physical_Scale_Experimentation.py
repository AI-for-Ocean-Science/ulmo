import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import time

# Variables for file paths
### Example: fn = r"C:\Users\Ben\Desktop\Prochaska Group\SSH Data\SSH_Data_Files\ssh_grids_v1812_1992100212.nc"
fn = "YOUR FILE PATH HERE"

# This is a function that takes in the file path, coordinates, and a plot size and returns a countoured Plate Caree map of a region specified by the coordinates and plot size.
def SSH_Map_Section(filepath,latx,lony,sqrsizeKm):

    ds = xr.open_dataset(filepath)

    SSH = ds['SLA'].mean(dim="Time").transpose() # Averages time to make the data 2D. Also transposed the axis because for some reason in the raw data they're flipped
    
    # Setting the x and y bounds for the plot
    lat = ds.variables['Latitude'][:]
    lon = ds.variables['Longitude'][:]

    # Setting up the plot
    fig = plt.figure(figsize=(8, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
   
    LatLonDeg = (sqrsizeKm / 110.574) / 2 # This is what will be used to set th eextent of the plot. The "/ 2" is because it adds/subtracts the "radius" from the center point
    coordinate = str("({0}°N,{1}°E)").format(latx,lony) # for formatting the center point to be put into the plot title
    squaresize = str("{0}x{1}Km").format(sqrsizeKm,sqrsizeKm) # for formatting the plot size to be put into the plot title

    ax.set_title("{0} SSH plot centered at {1}".format(squaresize,coordinate))
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.right_labels = False
    ax.set_extent([lony - LatLonDeg, lony + LatLonDeg, latx + LatLonDeg, latx - LatLonDeg], ccrs.PlateCarree()) # left lon bound, right lon bound, top lat bound, bottom lat bound

   # Making the Color Bar
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    CF = ax.contourf(lon, lat, SSH, transform=ccrs.PlateCarree()) # need this one for contour color
    cbar = plt.colorbar(CF,cax=cax)
    cbar.set_label('SSH (m)', rotation=270)


TimerStart = time.perf_counter()

### Example SSH_Map_Section(fn,40,-50,5000)
SSH_Map_Section(fn,,,)    

TimerStop = time.perf_counter() 
RunTime = TimerStop - TimerStart
print(("Run time: {:.2f} seconds").format(RunTime))



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    