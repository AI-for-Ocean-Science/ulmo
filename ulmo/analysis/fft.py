# imports
import numpy as np
from matplotlib import mlab
from scipy import ndimage as ni

import numpy as np
import numpy.fft as np_fft
from matplotlib import mlab
from scipy import ndimage as ni
from scipy import signal

import numpy as np

from IPython import embed

"""
Iterate through all cutouts within a file to calculate the average power spectrum, slope and intercept for two specified wavelength ranges in the zonal and meridional direction. Apply a hanning window or detrend/demean the data.  Save in a h5 file. 
"""
'''
start = time.time()

filename = input('Enter filename from s3 with cutouts: ')

new_filename = input('Enter filename with h5 extension to contain all spectral info: ')

apply_hanning_filter = input('To apply Hanning window, enter y : ')

# Hanning window or Detrend/Demean
'''

def process_preproc_file(pp_hf, dtdm=True, debug=False, key='valid'):
    """Calculate slopes for an input PreProc file

    Args:
        pp_hf (h5py.File): [description]
        dtdm (bool, optional): Detrend and demean? Defaults to True.
        debug (bool, optional): Defaults to False.
        key (str, optional): Dataset in the hdf5 file to analyze. Defaults to 'valid'.

    Returns:
        tuple: data1, data2, slopes, data4
            See code below for the specifics
    """

    # Load up
    print("Loading up the data...")
    if debug: 
        imgs = pp_hf[key][0:100, ...]
    else:
        imgs = pp_hf[key][()]
    print("Done loading the data...")
        
    num_of_cutouts = imgs.shape[0]
    print_out_list = np.arange(0, num_of_cutouts, 10000)
        

    # Initialize arrays
    data1 = np.zeros( (num_of_cutouts, 2, 32) )
    data2 = np.zeros( (num_of_cutouts, 32) )
    slopes = np.zeros( (num_of_cutouts, 6) )
    data4 = np.zeros( (num_of_cutouts, 4) )

    # Loop thru all cutouts
    print(f"Starting the loop of {num_of_cutouts} cutouts")
    for idx in range(num_of_cutouts):
        # image
        img = imgs[idx,0,...]

        # Do it
        zonal_results, merid_results = analyze_cutout(img, dtdm=dtdm)

        # assign values a place in new file
        data1[idx, 0, ...]    = zonal_results['psd_mean']
        data1[idx, 1, ...]    = merid_results['psd_mean']
        data2[idx, :]         = zonal_results['wavenumbers']
        slopes[idx, 0]         = zonal_results['slope_small']
        slopes[idx, 1]         = zonal_results['slope_large']
        slopes[idx, 2]         = zonal_results['slope_large_err']
        slopes[idx, 3]         = merid_results['slope_small']
        slopes[idx, 4]         = merid_results['slope_large']
        slopes[idx, 5]         = merid_results['slope_large_err']
        data4[idx, 0]         = zonal_results['intercept_small']
        data4[idx, 1]         = zonal_results['intercept_large']
        data4[idx, 2]         = merid_results['intercept_small']
        data4[idx, 3]         = merid_results['intercept_large']

        if idx in print_out_list:
            print('Currently at {} / {}.'.format( idx, num_of_cutouts))

    return data1, data2, slopes, data4

def analyze_cutout(img, dtdm=True):
    """Analyze an input cutout

    Args:
        img (np.ndarray): Input cutout
        dtdm (bool, optional): _description_. Defaults to True.

    Returns:
        tuple: zonal_results, merid_results
    """

    # call ffft
    zonal_results = fast_fft(array=img, dim=0, d=2000., Detrend_Demean=dtdm) 

    merid_results = fast_fft(array=img, dim=1, d=2000., Detrend_Demean=dtdm)

    return zonal_results, merid_results


def fast_fft( array, dim, d, small_range= [6000,15000], 
             large_range=[12000,50000], Detrend_Demean=False ):
   
    """ Fast- Fast Fourier Transform to calculate the power spectral density
   
    Parameters
    ----------
    array            (np.ndarray) : cutout
    dim              (float)      : Dimension
                                  along scan/ row (0) or along track / column (1)
    d                (int)        : sample spacing in meters
    small_range      (tuple)      : calculates spectral slope and intercept for smaller wavelengths
    large_range      (tuple)      : calculates spectral slope and intercept for larger wavelengths
    Detrend_Demean   (bool)       : if True, detrend and demean the array
   
    Returns
    -------
    results          (dict)
        psd_mean         (np.ndarray) : average power spectrum coefficients
        wavenumbers      (np.ndarray) : corresponding spatial frequencies
        slope_small      (float)      : slope of psd_mean within small wavelength range
        intercept_small  (float)      : intercept of (above)
        slope_large      (float)      : slope of psd_mean within large wavelength range
        slope_large_err  (float)      : slope of psd_mean within large wavelength range
        intercept_large  (float)      : intercept of (above)
    """   
    results = {}
   
    # find size of array along dimension of interest
    N = array.shape[ dim ]
    #test = np_fft.rfft(a=array, axis=dim)

    
    # Get the spectrum
    
    if Detrend_Demean:
        
        if dim == 1:
            
            #detrend
            detrend = signal.detrend( data=array, axis=0, type='linear')
            #demean
            demean  = detrend - np.mean( detrend, axis=0 )
            
            #fourier transform
            FFT = np_fft.rfftn( np.transpose(demean) )
            PSD = np.abs(FFT * np.conj(FFT))
            
            freq = np_fft.rfftfreq(N , d = d)
   
        else:
                             
            #detrend
            detrend = signal.detrend( data=array, axis=1, type='linear')
            #demean
            demean  = detrend - np.mean( detrend, axis=1 )
            
            #fourier transform
            FFT = np_fft.rfftn(demean)
            PSD = np.abs(FFT * np.conj(FFT))
            
            freq = np_fft.rfftfreq(N, d = d)

    else: #apply hanning window
                
        if dim == 1:
            
            hann_array = ni.convolve1d( array, np.hanning( array.shape[0], axis=0 ))
       
            #fourier transform
            FFT = np_fft.rfftn(hann_array)
            PSD = np.abs(FFT * np.conj(FFT))
            
            freq = np_fft.rfftfreq(N, d = d)
   
        else:
            
            hann_array = ni.convolve1d( array, np.hanning( array.shape[1], axis=1 ))
       
            #fourier transform
            FFT = np_fft.rfftn(hann_array)
            PSD = np.abs(FFT * np.conj(FFT))
            
            freq = np_fft.rfftfreq(N, d = d)


    #Now get average PSD spectrum with ensemble average.
 
    results['psd_mean'] = np.mean(PSD, 0)[1:]
 
 
    #Get the wavenumbers for this FFT.
    wavenumbers = freq[1:]

    #Finally calculate the slope for wavelength ranges specified or on default.
 
    # Apply median filter to signal
    psd_medf = ni.median_filter( results['psd_mean'], size=5 )
        
    # Determine the best fit to the specified range.
   
    ww_small = np.where( ( wavenumbers>(1/small_range[1])) & (
        wavenumbers<(1/small_range[0])))[0]
    ww_large = np.where( ( wavenumbers>(1/large_range[1])) & (
        wavenumbers<(1/large_range[0])))[0]
    
    pp_small, V_small = np.polyfit( np.log10(wavenumbers[ww_small]), 
                          np.log10(psd_medf[ww_small]),1, cov=True)
    pp_large, V_large = np.polyfit( np.log10(wavenumbers[ww_large]), 
                          np.log10(psd_medf[ww_large]),1, cov=True)

    results['wavenumbers'] = wavenumbers
    results['slope_small'] = pp_small[0]
    results['intercept_small'] = pp_small[1]
    
    results['slope_large'] = pp_large[0]
    results['slope_large_err'] = np.sqrt(V_large[0][0])
    results['intercept_large'] = pp_large[1]

    return results


def matlab_fast_fft( array, dim, d, small_range= [6000,15000], 
                    large_range=[12000,50000], Detrend_Demean=False ):
   
    """ Fast- Fast Fourier Transform to calculate the power spectral density
   
    Parameters
    ----------
    array            (np.ndarray) : cutout
    dim              (float)      : Dimension
                                  along scan/ row (0) or along track / column (1)
    d                (int)        : sample spacing in meters
    small_range      (tuple)      : calculates spectral slope and intercept for smaller wavelengths
    large_range      (tuple)      : calculates spectral slope and intercept for larger wavelengths
    Detrend_Demean   (bool)       : if True, detrend and demean the array
   
    Returns
    -------
    psd_mean         (np.ndarray) : average power spectrum coefficients
    wavenumbers      (np.ndarray) : corresponding spatial frequencies
    slope_small      (float)      : slope of psd_mean within small wavelength range
    intercept_small  (float)      : intercept of (above)
    slope_large      (float)      : slope of psd_mean within large wavelength range
    intercept_large  (float)      : intercept of (above)
    """   
   
    # find size of array along dimension of interest

    N = array.shape[ dim ]
    test = np_fft.rfft(a=array, axis=dim)
    L = test.shape[dim]

    
    # Get the spectrum
    
    if Detrend_Demean:
                     
        window = mlab.window_none                 
        if dim == 1:
        
            # Initialize array
            
            psd_array = np.empty( (L, array.shape[dim]) )
       
            for col in range(N):
                y = array[:, col]
                x = np.arange(0, y.shape[0], 1)
                pp = np.polyfit( x, y, 1)
                y_delin = y - np.polyval(pp, x) # delinearize
                psd_array[:, col], freqs = mlab.psd(y_delin, NFFT=N, Fs= 1./d, window=window, detrend='mean')
   
        else:
            # Initialize arrays
            
            psd_array = np.empty( (array.shape[dim], L ))
                             
            for row in range(N):
                y = array[row, :]
                x = np.arange(0, y.shape[0], 1)
                pp = np.polyfit( x, y, 1)
                y_delin = y - np.polyval(pp, x)
                psd_array[row,:], freqs = mlab.psd(y_delin, NFFT=N, Fs= 1./d, window=window, detrend='mean')

    else: #apply hanning window
                     
        window = mlab.window_hanning                 
        if dim == 1:
       
            for col in range(N):
                y = array[:, col]
                psd_array[:, col], freqs = mlab.psd(y, NFFT=N, Fs= 1./2000, window=window)
   
        else:
            for row in range(N):
                y = array[row, :]
                psd_array[row,:], freqs = mlab.psd(y, NFFT=N, Fs= 1./2000, window=window)


    #Now get average PSD spectrum with ensemble average.
 
    if dim == 1:
        psd_mean = np.mean(psd_array, 1)[1:]
   

    else:
        psd_mean = np.mean(psd_array, 0)[1:] 

 
    #Get the wavenumbers for this FFT.
 
    wavenumbers = freqs[1:]


 
    #Finally calculate the slope for wavelength ranges specified or on default.
 
    # Apply median filter to signal
    psd_medf = ni.median_filter( psd_mean, size=5 )
        
    # Determine the best fit to the specified range.
   
    ww_small = np.where( ( wavenumbers>(1/small_range[1])) & (wavenumbers<(1/small_range[0])))[0]
    ww_large = np.where( ( wavenumbers>(1/large_range[1])) & (wavenumbers<(1/large_range[0])))[0]
    
    pp_small = np.polyfit( np.log10(wavenumbers[ww_small]), np.log10(psd_medf[ww_small]),1)
    pp_large = np.polyfit( np.log10(wavenumbers[ww_large]), np.log10(psd_medf[ww_large]),1)
    
    slope_small = round(pp_small[0], 2)
    intercept_small = round(pp_small[1], 2)
    
    slope_large = round(pp_large[0], 2)
    intercept_large = round(pp_large[1], 2)

    return psd_mean, wavenumbers, slope_small, intercept_small, slope_large, intercept_large

