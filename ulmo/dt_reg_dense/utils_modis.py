import torch
from torch.utils.data import DataLoader, Dataset
import h5py 
import pandas
import random
import numpy as np

class MODISDTDataset(Dataset):
    """
    LLCFS Dataset used for the training of the regression model.
    """
    def __init__(self, feature_path, label_path, file_id, data_key='train'):
        self.data_key = data_key
        self.file_id = file_id
        self._open_file(feature_path, label_path)
        train_index, valid_index = self._train_valid_split()
        if data_key == 'train':
            self.data_index_list = train_index
        else:
            self.data_index_list = valid_index
        
    def _open_file(self, feature_path, label_path):
        main_table = pandas.read_parquet(label_path)
        self.pp_idx_array = main_table[main_table['pp_file'] == self.file_id].pp_idx.values.astype(np.int32)
        self.dt_array = main_table[main_table['pp_file'] == self.file_id].DT40.values.astype(np.float32)
        self.feature = h5py.File(feature_path, 'r')['valid']
        
    def _train_valid_split(self):
        num_samples = self.pp_idx_array.shape[0]
        valid_samples = num_samples // 10
        train_samples = num_samples - valid_samples
        index_list = list(range(num_samples))
        random.seed(0)
        random.shuffle(index_list)
        train_index = index_list[:train_samples]
        valid_index = index_list[train_samples:]
        return train_index, valid_index
        
    def __len__(self):
        num_samples = len(self.data_index_list)
        return num_samples

    def __getitem__(self, global_idx):     
        data_index = self.data_index_list[global_idx]
        pp_idx = self.pp_idx_array[data_index]
        dt = self.dt_array[data_index]
        feature = self.feature[pp_idx]
        return feature, dt
    
def modisdt_loader(feature_path, label_path, file_id, batch_size, train_flag='train'):
    """
    This is a function used to create a LLCFS data loader.
    
    Args:
        feuture_path: (str) path of feature file;
        label_path: (str) path of label file;
        file_id: (str) id of the file offerering the latents;
        batch_size: (int) batch size;
        train_flag: (str) flag of train or valid mode;
        
    Returns:
        loader: (Dataloader) MODIS DT Dataloader.
    """
    modisdt_dataset = MODISDTDataset(
        feature_path, 
        label_path, 
        file_id, 
        data_key=train_flag,
    )
    loader = torch.utils.data.DataLoader(
        modisdt_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=False
    )
    return loader