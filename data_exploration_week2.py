import numpy as np
import xarray
import pandas
import matplotlib.pyplot as plt

data_xr = xarray.open_dataset("/Volumes/Aqua-1/Hackathon/daily/l3s_fields/2019/009/20190109120000-STAR-L3S_GHRSST-SSTsubskin-LEO_Daily-ACSPO_V2.80-v02.0-fv01.0.nc")

print(data_xr)

# kind of a random plot, just experimenting with plotting two values for the data

plt.scatter(data_xr['sea_surface_temperature'], data_xr['wind_speed'])
plt.xlabel('sea surface temperature')
plt.ylabel('wind speed')
plt.show()

# plots latitude and longitude with sea surface temperature in a dotted plot format

data_wo_time = data_xr.squeeze(dim = 'time')
data_wo_time['sea_surface_temperature'].plot.pcolormesh()
plt.show()

# same data and plot with contour instead of points

data_wo_time['sea_surface_temperature'].plot.contourf()
plt.show()

# repeats the plotting pattern with another xarray in the data set

data_xr_day_2 = xarray.open_dataset("/Volumes/Aqua-1/Hackathon/daily/l3s_fields/2019/010/20190110120000-STAR-L3S_GHRSST-SSTsubskin-LEO_Daily-ACSPO_V2.80-v02.0-fv01.0.nc")

data_wo_time_day2['sea_surface_temperature'].plot.pcolormesh()
plt.show()

# This subsets the small area I selected, and combined the data sets (which I didn't end up using)
data_xr_subset = data_xr.sel(lat=slice(40, 20),lon=slice(-75,-50))

data_xr_day2_subset = data_xr_day2.sel(lat=slice(40, 20),lon=slice(-75,-50))

combined_dataset = xarray.concat([data_xr_subset, data_xr_day2_subset], dim='time')

# start_time = '2015-12-23T12:00:00'
# end_time = '2015-12-24T12:00:00'
# time_range = combined_dataset.sel(time=slice(start_time, end_time))

# In this section, I selected a small area of the plot and tried to plot them in a time-series format,
# I was able to make them side-by-side, yet would like to be able to see this in kind of a movie format

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

data_xr_subset = data_xr_subset.squeeze(dim = 'time')
data_xr_subset['sea_surface_temperature'].plot.pcolormesh(ax=ax1)
time_values = data_xr_subset['time'].values
ax1.set_title(f'Time: {time_values}')

data_xr_day2_subset = data_xr_day2_subset.squeeze(dim = 'time')
data_xr_day2_subset['sea_surface_temperature'].plot.pcolormesh(ax=ax2)
time_values = data_xr_day2_subset['time'].values
ax2.set_title(f'Time: {time_values}')

plt.show()


