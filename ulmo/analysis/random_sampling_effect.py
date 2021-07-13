# Create 100 copies of the same image
# Add noise sampled from same distribution to each
# Answer the question: is ulmo ultra-sensitive to noise? 

import h5py
import numpy as np

pp_idx = 609719

with h5py.File('/home/jovyan/ulmo/ulmo/notebooks/LLC_modis2012_test_preproc.h5', 'r') as f:
    img = f['valid'][pp_idx, 0, ...]

# make 100 copies
with h5py.File('/home/jovyan/ulmo/ulmo/notebooks/hundred_imgs.h5', 'w') as h:
    data = h.create_dataset('valid', (100, 1, 64, 64))
    
    
    for i in range(100): 
        
        #add noise
        noise = np.random.normal(0., 0.12, (64,64))
        h['valid'][i, 0, ...] = img + noise
        
        # demean
        image = h['valid'][i, 0, ...]
        mean = np.mean(image)
        h['valid'][i, 0, ...] = image - mean*np.ones_like(image)