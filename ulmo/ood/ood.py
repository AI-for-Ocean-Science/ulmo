import os
import time
import sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from tqdm.auto import tqdm


class ProbabilisticAutoencoder:
    """A probabilistic autoencoder (see arxiv.org/abs/2006.05479)."""
    def __init__(self, autoencoder, flow, data,  
                 autoencoder_optimizer=None, flow_optimizer=None, 
                 valid_fraction=0.05, logdir='./', device=None):
        """
        Parameters
            autoencoder: ulmo.models.Autoencoder
                Autoencoder model for dimensionality reduction
            flow: ulmo.models.ConditionalFlow
                Flow model for likelihood estimation
            data: np.ndarray
                Training data for out-of-distribution detection,
                should be of shape (*, D) for vector data or
                (*, C, H, W) for image data
            autoencoder_optimizer: torch.optim.Optimizer
                Optimizer for autoencoder model
            flow_optimizer: torch.optim.Optimizer
                Optimizer for flow model
            valid_fraction: float
                Fraction of data to hold out for validation
            logdir: str
                Location to store logs and models
            device: torch.device
                Device to use for training and inference
        """
        self.autoencoder = autoencoder
        self.flow = flow
        self.logdir = logdir
        self.valid_fraction = valid_fraction
        self.flow_filepath = os.path.join(logdir, 'flow.pt')
        self.autoencoder_filepath = os.path.join(logdir, 'autoencoder.pt')
        self.best_ae_val_loss = np.inf
        self.best_flow_val_loss = np.inf
        assert flow.dim = autoencoder.latent_dim
        
        if autoencoder_optimizer is None:
            autoencoder_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=2.5e-4)
        if flow_optimizer is None:
            flow_optimizer = torch.optim.AdamW(flow.parameters(), lr=2.5e-4)
        
        self.ae_opt = autoencoder_optimizer
        self.flow_opt = flow_optimizer
        
        n = data.shape[0]
        idx = sklearn.utils.shuffle(np.arange(n))
        m = int(self.valid_fraction * n)
        valid_idx = idx[:m]
        train_idx = idx[m:]
        
        self.data = {'train_data': data[train_idx],
                     'valid_data': data[valid_idx]}
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.device = device
        
    def make_loaders(self, train_data, valid_data, batch_size, drop_last=True):
        train_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_data).float())
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        valid_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(valid_data).float())
        valid_loader = torch.utils.data.DataLoader(
            valid_dset, batch_size=batch_size, shuffle=False, drop_last=drop_last)

        return train_loader, valid_loader
    
    def save_autoencoder(self):
        torch.save(self.autoencoder.state_dict(), self.autoencoder_filepath)
        
    def save_flow(self):
        torch.save(self.flow.state_dict(), self.flow_filepath)
        
    def train_model(self, model_type, n_epochs, batch_size):
        try:
            model_type = model_type.strip().lower()
            if model_type == 'autoencoder':
                model = self.autoencoder
                optimizer = self.ae_opt
                save_model = self.save_autoencoder
                train_loader, valid_loader = self.make_loaders(
                    self.data['train_data'], self.data['valid_data'], batch_size)
                best_val_loss = self.best_ae_val_loss
                print(f"Training on {self.data['train_data'].shape[0]:,d} samples. "
                      f"Validating on {self.data['valid_data'].shape[0]:,d} samples.")
                
            elif model_type == 'flow':
                model = self.flow
                optimizer = self.flow_opt
                save_model = self.save_flow
                train_loader, valid_loader = self.make_loaders(
                    self.data['train_latents'], self.data['valid_latents'], batch_size)
                best_val_loss = self.best_flow_val_loss
                print(f"Training on {self.data['train_latents'].shape[0]:,d} samples. "
                      f"Validating on {self.data['valid_latents'].shape[0]:,d} samples.")
            else:
                raise ValueError(f"Model type {model_type} unknown.")
                
            model.train()
            global_step = 0
            total_loss = 0

            # Train
            pbar = tqdm(range(n_epochs))
            for epoch in pbar:
                for data in train_loader:
                    optimizer.zero_grad()
                    loss = model(data.to(self.device))
                    loss.backward()
                    total_loss += loss.item()
                    optimizer.step()
                    global_step += 1

                # Evaluate
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    for i, data in enumerate(valid_loader):
                        loss = model(data.to(self.device))
                        total_loss += loss.item()
                val_loss = total_loss / float(i+1)
                if val_loss < best_val_loss:
                    save_model()
                    best_val_loss = val_loss
                    if model_type == 'autoencoder':
                        self.best_ae_val_loss = val_loss
                    elif model_type == 'flow':
                        self.best_flow_val_loss = val_loss
                pbar.set_description(f"Validation Loss: {val_loss:.3f}")
                self.model.train()
                
        except KeyboardInterrupt:
            save = input("Training stopped. Save model (y/n)?").strip().lower() == 'y'
            if save:
                save_model()
                print("Model saved.")
                
    def store_latents(self):
        self.autoencoder.eval()
        train_loader, valid_loader = self.make_loaders(
            self.data['train_data'], self.data['valid_data'], 64, drop_last=False)
        
        with torch.no_grad():
            z = [self.autoencoder.encode(data.to(self.device)).detach().cpu().numpy()
                 for data in train_loader]
            data['train_latents'] = np.concatenate(z)
            z = [self.autoencoder.encode(data.to(self.device)).detach().cpu().numpy()
                 for data in valid_loader]
            data['valid_latents'] = np.concatenate(z)
            
    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
            return x
        elif isinstance(x, torch.Tensor):
            return x
        else:
            t = type(x)
            raise ValueError(f"Type {t} not supported.")
            
    def to_array(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            return x
        elif isinstance(x, np.ndarray):
            return x
        else:
            t = type(x)
            raise ValueError(f"Type {t} not supported.")
        
    def encode(self, x):
        x = self.to_tensor(x)
        z = self.autoencoder.encode(x)
        return z
    
    def decode(self, z):
        z = self.to_tensor(z)
        x = self.autoencoder.decode(z)
        return x
    
    def reconstruct(self, x):
        x = self.to_tensor(x)
        rx = self.autoencoder.reconstruct(x)
        return rx
    
    def log_prob(self, x):
        x = self.to_tensor(x)
        z = self.encode(x)
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float().to(self.device)
        log_prob = self.flow.log_prob(z)
        return log_prob
    
    def train(self, autoencoder_epochs=20, autoencoder_batch_size=128,
              flow_epochs=20, flow_batch_size=64):
        
        self.train_model('autoencoder', autoencoder_epochs, autoencoder_batch_size)
        self.store_latents()
        self.train_model('flow', flow_epochs, flow_batch_size)
        