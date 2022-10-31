import h5py
import numpy as np
from sklearn.model_selection import train_test_split


filepath = "LLC_uniform144_test_preproc.h5"

# Right now we want to read the file, split it into 'train' and 'valid'
# then save them as a new file

with h5py.File(filepath, 'r') as f:
    print(f.keys())   # prints <KeysViewHDF5 ['valid', 'valid_metadata']>
    len_valid = f['valid'].shape[0]
    print(len_valid)
    llc_modis_uniform = f['valid'][:] # Pretty sure this is pulling the file into mem

print("suck my nuts please")
    
llc_modis_uniform_train, llc_modis_uniform_test = train_test_split(
     llc_modis_uniform, test_size=0.1, random_state=0)

print("awawawawaaa")

filepath_split = 'LLC_uniform_test_preproc_split.h5'
with h5py.File(filepath_split, 'w') as f:
    f.create_dataset('train', data=llc_modis_uniform_train)
    f.create_dataset('valid', data=llc_modis_uniform_test)

print("wah")

del(llc_modis_uniform)
del(llc_modis_uniform_train)
del(llc_modis_uniform_test)

with h5py.File('LLC_uniform_test_preproc_split.h5', 'r') as fr:
    # print dataset shape and values
    print(fr['train'].shape, fr['train'].dtype)
    print(fr['valid'].shape, fr['valid'].dtype)
