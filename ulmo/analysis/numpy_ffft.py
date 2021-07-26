import numpy as np
import numpy.fft as fft
from matplotlib import mlab
from scipy import ndimage as ni
from scipy import signal

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
        
        if dim == 1:
            
            #detrend
            detrend = signal.detrend( data=array, axis=0, type='linear')
            #demean
            demean  = detrend - np.mean( detrend, axis=0 )
            
            #fourier transform
            FFT = fft.rfftn( np.transpose(demean) )
            PSD = np.abs(FFT * np.conj(FFT))
            
            freq = fft.rfftfreq(N , d = d)
   
        else:
                             
            #detrend
            detrend = signal.detrend( data=array, axis=1, type='linear')
            #demean
            demean  = detrend - np.mean( detrend, axis=1 )
            
            #fourier transform
            FFT = fft.rfftn(demean)
            PSD = np.abs(FFT * np.conj(FFT))
            
            freq = fft.rfftfreq(N, d = d)

    else: #apply hanning window
                
        if dim == 1:
            
            hann_array = ni.convolve1d( array, np.hanning( array.shape[0], axis=0 ))
       
            #fourier transform
            FFT = fft.rfftn(demean)
            PSD = np.abs(FFT * np.conj(FFT))
            
            freq = fft.rfftfreq(N, d = d)
   
        else:
            
            hann_array = ni.convolve1d( array, np.hanning( array.shape[1], axis=1 ))
       
            #fourier transform
            FFT = fft.rfftn(demean)
            PSD = np.abs(FFT * np.conj(FFT))
            
            freq = fft.rfftfreq(N, d = d)


    #Now get average PSD spectrum with ensemble average.
 
    psd_mean = np.mean(PSD, 0)[1:]
 
 
    #Get the wavenumbers for this FFT.
 
    wavenumbers = freq[1:]


 
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