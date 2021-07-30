# Add SST-dependent noise to 'LLC_modis2012_SST_noise_demean_preproc.h5' file
# The file is a copied verision of 'LLC_modis2012_test_preproc.h5' file

import h5py
import numpy as np

with h5py.File( '/home/jovyan/LLC_modis2012_noise_track_preproc.h5', 'r+') as f:
    mdata = f['valid_metadata'][()]
    mean_T = mdata[:,7]
     
    # loop through all cutouts
    for idx in range( mdata.shape[0] ):
        tmp = float( mean_T[idx].decode() )
        
        #sig = 0.031 + 0.0048*tmp 
        sig = 0.038 + 0.0054*tmp
        
        #divide by two: modis noise has to be averaged
        noise = np.random.normal(0., sig/2, (64,64))
        f['valid'][idx, 0, ...] = f['valid'][idx, 0, ...] + noise
        
        # demean
        img = f['valid'][idx, 0, ...]
        mean = np.mean( img )
        f['valid'][idx, 0, ...] = img - mean*np.ones_like( img )
    
