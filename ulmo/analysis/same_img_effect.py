# Create 100 copies of the same image
# Answer the question: how variable is ulmo with the same image? 

import h5py
import numpy as np

pp_idx = 609719

with h5py.File('/home/jovyan/ulmo/ulmo/notebooks/LLC_modis2012_test_preproc.h5', 'r') as f:
    img = f['valid'][pp_idx, 0, ...]

# make 100 copies
with h5py.File('/home/jovyan/ulmo/ulmo/notebooks/hundred_imgs_same.h5', 'w') as h:
    data = h.create_dataset('valid', (100, 1, 64, 64))
     
        
    #add noise
    noise = np.random.normal(0., 0.12, (64,64))
    image = img + noise
        
    # demean
    mean = np.mean(image)
    h['valid'][:, 0, ...] = image - mean*np.ones_like(image)