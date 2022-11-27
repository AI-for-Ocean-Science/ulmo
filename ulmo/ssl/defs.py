""" Define data model, options, etc. for SSL models and analysis"""
import numpy as np

# SSL options
ssl_opt_dmodel = {
    'print_freq': dict(dtype=(int, np.integer),
                help='How often to print to screen'),
    'save_freq': dict(dtype=(int, np.integer),
                help='How often to save the model [unit=epochs]'),
    'valid_freq': dict(dtype=(int, np.integer),
                help='How often to calculate the validation loss [unit=epochs]'),
    'batch_size': dict(dtype=(int, np.integer),
                help='Batch size.  The same value is used for train, validation, and evaluation'),
    'num_workers': dict(dtype=(int, np.integer),
                help='Workers in the DataLoader.  See train_util.py'),
    'epochs': dict(dtype=(int, np.integer),
                help='Number of epochs for training.'),
    'learning_rate': dict(dtype=float,
                help='Learning rate parameter'),
    'lr_decay_epochs': dict(dtype=list,
                help='Learning rate decay parameters'),
    'lr_decay_rate': dict(dtype=float,
                help='Learning rate decay rate'),
    'feat_dim': dict(dtype=(int, np.integer),
                help='Dimensionality of latent space'),
    'weight_decay': dict(dtype=float,
                help='Decay parameter in optimizer'),
    'momentum': dict(dtype=float,
                help='Momentum parameter in optimizer'),
    'ssl_model': dict(dtype=str,
                help='Model for SSL'),
    'ssl_method': dict(dtype=str,
                help='Method for SSL'),
    'model_root': dict(dtype=str,
                help='Root name of the model.  Used for the file tree'),
    'model_name': dict(dtype=str,
                help='Name of the specific model.  Based on parameters'),
    'images_file': dict(dtype=str,
                help='Name of the images file, likely hdf5'),
    'data_folder': dict(dtype=str,
                help='Path to the images_file'),
    'model_folder': dict(dtype=str,
                help='Full path to folder for saving models'),
    'latents_folder': dict(dtype=str,
                help='Full path to folder for saving latents files'),
    'train_key': dict(dtype=str,
                help='Dataset for training'),
    'valid_key': dict(dtype=str,
                help='Dataset for validation'),
    's3_outdir': dict(dtype=str,
                help='s3 bucket+path for model output'),
    'temp': dict(dtype=float,
                help='Temperature parameter in Loss function'),
    'cosine': dict(dtype=bool,
                help='??'),
    'syncBN': dict(dtype=bool,
                help='Enable synchronization of Batch Normalization?'),
    'warm': dict(dtype=bool,
                help='??'),
    'warmup_from': dict(dtype=float,
                help='??'),
    'warmup_to': dict(dtype=float,
                help='??'),
    'warm_epochs': dict(dtype=(int, np.integer),
                help='Number of epochs for warming up'),
    'trial': dict(dtype=(int, np.integer),
                help='Index for labeling the model'),
    'random_jitter': dict(dtype=list,
                help='x,y jitter parameters'),
    'gauss_noise': dict(dtype=float,
                help='RMS for injected Gaussian noise'),
    'modis_data': dict(dtype=bool,
                help='MODIS data?'),
    'cuda_use': dict(dtype=bool,
                help='Use CUDA in the analysis, if possible'),
}

# UMAP DT CUTS
umap_DT = {}

umap_DT['DT0'] = (0.25, 0.25)
umap_DT['DT1'] = (0.75, 0.25)
umap_DT['DT15'] = (1.25, 0.25)
umap_DT['DT2'] = (2.0, 0.5)
umap_DT['DT4'] = (3.25, 0.75)
umap_DT['DT5'] = (4.0, -1)
umap_DT['all'] = None
umap_DT['DT10'] = (1.0, 0.05)

# UMAP alpha CUTS
umap_alpha = {}

umap_alpha['a0'] = (-0.25, 0.25)
umap_alpha['a1'] = (-0.75, 0.25)
umap_alpha['a15'] = (-1.25, 0.25)
umap_alpha['a2'] = (-1.75, 0.25)
umap_alpha['a25'] = (-2.25, 0.25)
umap_alpha['a3'] = (-2.75, 0.25)
umap_alpha['a4'] = (-3.0, -1)
umap_alpha['all'] = None