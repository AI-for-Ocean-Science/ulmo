""" Package to explore an input image """
import os
import numpy as np

import torch

from ulmo.ssl import latents_extraction
from ulmo import io as ulmo_io

def get_latents(img:np.ndarray, 
                model_file:str, 
                opt:ulmo_io.Params):

    # Build the SSL model
    model_base = os.path.basename(model_file)
    if not os.path.isfile(model_base):
        ulmo_io.download_file_from_s3(model_base, model_file)
    else:
        print(f"Using already downloaded {model_base} for the model")

    # DataLoader
    dset = torch.utils.data.TensorDataset(torch.from_numpy(img).float())
    data_loader = torch.utils.data.DataLoader(
        dset, batch_size=1, shuffle=False, collate_fn=None,
        drop_last=False, num_workers=1)

    # Time to run
    latents = latents_extraction.model_latents_extract(
        opt, 'None', 'valid', 
        model_base, None, None,
         loader=data_loader)

    # Return
    return latents
    