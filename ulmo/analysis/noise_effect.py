# create 60 copies of same cutout image
# to answer the question : How does LL vary with noise?

import h5py
import numpy as np

pp_idx = 609719

with h5py.File('/home/jovyan/ulmo/ulmo/notebooks/LLC_modis2012_test_preproc.h5', 'r') as f:
    img = f['valid'][pp_idx, 0, ...]

# Add noise to 30 images
# 0 to 0.3 with step 0.01

with h5py.File('/home/jovyan/ulmo/ulmo/notebooks/sixty_imgs.h5', 'w') as h:
    data = h.create_dataset('valid', (60, 1, 64, 64))
    
    for i in range(30): 
        
        sig = 0.01*i
        noise = np.random.normal(0., sig, (64,64))
        h['valid'][i, 0, ...] = img + noise
        
        image = h['valid'][i, 0, ...]
        mean = np.mean(image)
        h['valid'][i, 0, ...] = image - mean*np.ones_like(image)

# Repeat
    for j in range(30): 
        
        sig = 0.01*j
        noise = np.random.normal(0., sig, (64,64))
        h['valid'][j + 30, 0, ...] = img + noise

        image = h['valid'][j, 0, ...]
        mean = np.mean(image)
        h['valid'][j + 30, 0, ...] = image - mean*np.ones_like(image)
        
