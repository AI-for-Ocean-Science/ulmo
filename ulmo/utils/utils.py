""" Custom Dataset for HDF5 files. """
import h5py
import numpy as np
from torch.utils import data
from torch.utils.data.dataloader import default_collate

class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Parameters:
        file_path: Path to the HDF5 file.
        dataset_names: List of dataset names to gather. 
            Objects will be returned in this order.
        return_mask: Return the mask on a call to __getitem__?
    """
    def __init__(self, file_path, partition, 
                 return_mask:bool=False,
                 mask_partition:str='masks'):
        super().__init__()
        self.file_path = file_path
        self.partition = partition
        self.meta_dset = partition + '_metadata'
        # Mask
        self.return_mask = return_mask
        self.mask_partition = mask_partition
        # s3 is too risky and slow here
        self.h5f = h5py.File(file_path, 'r')

    def __len__(self):
        return self.h5f[self.partition].shape[0]
    
    def __getitem__(self, index):
        data = self.h5f[self.partition][index]
        #if self.meta_dset in self.h5f.keys():
        #    metadata = self.h5f[self.meta_dset][index]
        #else:
        metadata = None

        if self.return_mask:
            mask = self.h5f[self.mask_partition][index]
            return data, mask
        else:
            return data, metadata
    

def id_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[0])
        ids.append(_batch[1])
    return default_collate(new_batch), np.array(ids)
    

def get_n_params(model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in trainable])
    return n_params


def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (0.5)


def get_quantiles(x):
    rank = np.searchsorted(sorted(x), x)
    quantile = rank / len(rank)
    return quantile
