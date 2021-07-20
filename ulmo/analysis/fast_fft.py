import numpy as np
import numpy.fft as fft
from matplotlib import mlab
from scipy import ndimage as ni

def fast_fft( array, dim, d, wavelength_range, Detrend_Demean=False ):
   
    """ Fast- Fast Fourier Transform to calculate the power spectral density
   
    Parameters
    ----------
    array            (np.ndarray) : cutout
    dim              (float)      : Dimension
                                  along scan/ row (0) or along track / column (1)
    d                (int)        : sample spacing in meters
    wavelength_range (tuple)      : if specified, calculates spectral slope and intercept
    Detrend_Demean   (bool)       : if True, detrend and demean the array
   
    Returns
    -------
    psd_mean         (np.ndarray) : average power spectrum coefficients
    wavenumbers      (np.ndarray) : corresponding spatial frequencies
    slope            (float)      : slope of psd_mean within wavelength range
    intercept        (float)      : intercept of (above)
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


 
    #Finally calculate the slope if a wavelength range has been specified.
 
    if wavelength_range==None:
        slope = nan
        intercept = nan
   
    else:
        
        # Apply median filter to signal
        psd_medf = ni.median_filter( psd_mean, size=3 )
        
        # Determine the best fit to the specified range.
   
        ww = np.where( ( wavenumbers>(1/wavelength_range[1])) & (wavenumbers<(1/wavelength_range[0])))[0]
        pp = np.polyfit( np.log10(wavenumbers[ww]), np.log10(psd_medf[ww]),1)
        slope = round(pp[0], 2)
        intercept = round(pp[1], 2)

    return psd_mean, wavenumbers, slope, intercept