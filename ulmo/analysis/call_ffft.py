# imports
import h5py
from ulmo.analysis import numpy_ffft as ffft
from ulmo import io as ulmo_io
import numpy as np
import time

"""
Iterate through all cutouts within a file to calculate the average power spectrum, slope and intercept for two specified wavelength ranges in the zonal and meridional direction. Apply a hanning window or detrend/demean the data.  Save in a h5 file. 
"""
start = time.time()

filename = input('Enter filename from s3 with cutouts: ')

new_filename = input('Enter filename with h5 extension to contain all spectral info: ')

apply_hanning_filter = input('To apply Hanning window, enter y : ')

# Hanning window or Detrend/Demean

if apply_hanning_filter == 'y': 
    dtdm = False
else: 
    dtdm = True

# Open file and Loop thru all cutouts
debug = False

# if with s3
#with ulmo_io.open( filename , 'rb') as f:
pp_hf = h5py.File(filename, 'r')

if debug: 
    imgs = pp_hf['valid'][0:100, ...]
else:
    imgs = pp_hf['valid'][()]

pp_hf.close()
    
num_of_cutouts = imgs.shape[0]
print_out_list = np.arange(0, num_of_cutouts, 10000)
    

# Initialize arrays
data1 = np.zeros( (num_of_cutouts, 2, 32) )
data2 = np.zeros( (num_of_cutouts, 32) )
data3 = np.zeros( (num_of_cutouts, 4) )
data4 = np.zeros( (num_of_cutouts, 4) )

# Loop thru all cutouts
for idx in range( num_of_cutouts ):

    # image
    img = imgs[idx,0,...]

    # call ffft
    zonal_psd, freq, zonal_slope_small, zonal_intercept_small, zonal_slope_large, zonal_intercept_large = ffft.fast_fft(array=img, dim=0, d=2000., Detrend_Demean=dtdm ) 

    merid_psd, freq, merid_slope_small, merid_intercept_small, merid_slope_large, merid_intercept_large = ffft.fast_fft(array=img, dim=1, d=2000., Detrend_Demean=dtdm )
    

    # assign values a place in new file
    data1[idx, 0, ...]    = zonal_psd
    data1[idx, 1, ...]    = merid_psd
    data2[idx, :]         = freq
    data3[idx, 0]         = zonal_slope_small
    data3[idx, 1]         = zonal_slope_large
    data3[idx, 2]         = merid_slope_small
    data3[idx, 3]         = merid_slope_large
    data4[idx, 0]         = zonal_intercept_small
    data4[idx, 1]         = zonal_intercept_large
    data4[idx, 2]         = merid_intercept_small
    data4[idx, 3]         = merid_intercept_large

    if idx in print_out_list:

        now = time.time() - start
        print('Currently at {} / {}. Time taken: {} '.format( idx, num_of_cutouts, now))

# Create new file, name, and its datasets
with h5py.File( new_filename, 'w' ) as g:
        
    g.create_dataset( 'spectra', data = data1)
    g.create_dataset( 'wavenumbers', data = data2 )
    g.create_dataset( 'slopes', data = data3)
    g.create_dataset( 'intercepts', data = data4)
    
end = time.time()
print('Total time: {} '.format(end - start))
    
    

