import numpy as np
import numpy.fft as fft
from matplotlib import mlab
from scipy import ndimage as ni

def fast_fft( array, dim, d, small_range= [6000,15000], large_range=[12000,50000], Detrend_Demean=False ):
   
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
    test = fft.rfft(a=array, axis=dim)
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