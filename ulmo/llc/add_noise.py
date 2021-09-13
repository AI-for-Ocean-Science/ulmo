# Add SST-dependent noise to 'LLC_modis2012_SST_noise_demean_preproc.h5' file
# The file is a copied verision of 'LLC_modis2012_test_preproc.h5' file

import h5py
import numpy as np


modis = False

if modis: 

    with h5py.File( '/home/jovyan/ulmo/ulmo/notebooks/LLC_modis2012_SST_noise_demean_preproc.h5', 'r+') as f:
        mdata = f['valid_metadata'][()]
        mean_T = mdata[:,7]
     
        # loop through all cutouts
        for idx in range( mdata.shape[0] ):
            tmp = float( mean_T[idx].decode() )
            sig = 0.031 + 0.0048*tmp
            noise = np.random.normal(0., sig, (64,64))
            f['valid'][idx, 0, ...] = f['valid'][idx, 0, ...] + noise
        
            # demean
            img = f['valid'][idx, 0, ...]
            mean = np.mean( img )
            f['valid'][idx, 0, ...] = img - mean*np.ones_like( img )
    
else: 
    
    with h5py.File( '/home/jovyan/LLC_uniform_viirs_test_preproc.h5', 'r+') as f:
        
        imgs = f['valid'][()]
     
        # loop through all cutouts
        for idx in range( imgs.shape[0] ):
            sig = 0.039
            noise = np.random.normal(0., sig, (64,64))
            imgs[idx, 0, ...] = imgs[idx, 0, ...] + noise
        
            # demean
            img = imgs[idx, 0, ...]
            mean = np.mean( img )
            f['valid'][idx, 0, ...] = img - mean*np.ones_like( img )    
