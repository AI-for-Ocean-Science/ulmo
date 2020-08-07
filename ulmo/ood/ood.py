import os
import time
import sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler


class ProbabilisticAutoencoder:
    """A probabilistic autoencoder (see arxiv.org/abs/2006.05479)."""
    def __init__(self, autoencoder, flow, data, valid_fraction=0.05, 
                 logdir='./', device=None):
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
            valid_fraction: float
                Fraction of data to hold out for validation
            logdir: str
                Location to store logs and models
            device: torch.device
                Device to use for training and inference
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == torch.device('cuda'):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
        self.device = device
        self.autoencoder = autoencoder.to(device)
        self.flow = flow.to(device)
        self.logdir = logdir
        self.valid_fraction = valid_fraction
        self.autoencoder_filepath = os.path.join(logdir, 'autoencoder.pt')
        self.flow_filepath = os.path.join(logdir, 'flow.pt')
        self.up_to_date_latents = False
        assert flow.dim == autoencoder.latent_dim
        
        n = data.shape[0]
        idx = sklearn.utils.shuffle(np.arange(n))
        m = int(self.valid_fraction * n)
        valid_idx = idx[:m]
        train_idx = idx[m:]
        
        self.data = {
            'train_data': data[train_idx],
            'valid_data': data[valid_idx]}
        self.best_valid_loss = {
            'flow': np.inf,
            'autoencoder': np.inf}
        
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
        
    def _train_module(self, module, n_epochs, batch_size, lr):
        try:
            module = module.strip().lower()
            if module == 'autoencoder':
                model = self.autoencoder
                save_model = self.save_autoencoder
                train_loader, valid_loader = self.make_loaders(
                    self.data['train_data'], self.data['valid_data'], batch_size)
                print(f"Training on {self.data['train_data'].shape[0]:,d} samples. "
                      f"Validating on {self.data['valid_data'].shape[0]:,d} samples.")
                self.up_to_date_latents = False
                
            elif module == 'flow':
                model = self.flow
                save_model = self.save_flow
                train_loader, valid_loader = self.make_loaders(
                    self.data['norm_train_latents'], self.data['norm_valid_latents'], batch_size)
                print(f"Training on {self.data['norm_train_latents'].shape[0]:,d} samples. "
                      f"Validating on {self.data['norm_valid_latents'].shape[0]:,d} samples.")
            else:
                raise ValueError(f"Module {module} unknown.")
            
            model.train()
            global_step = 0
            total_loss = 0
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            
            # Train
            pbar = tqdm(range(n_epochs))
            for epoch in pbar:
                for data in train_loader:
                    optimizer.zero_grad()
                    loss = model(data[0].to(self.device))
                    loss.backward()
                    total_loss += loss.item()
                    optimizer.step()
                    global_step += 1

                # Evaluate
                model.eval()
                with torch.no_grad():
                    total_loss = 0
                    for i, data in enumerate(valid_loader):
                        loss = model(data[0].to(self.device))
                        total_loss += loss.item()
                valid_loss = total_loss / float(i+1)
                if valid_loss < self.best_valid_loss[module]:
                    save_model()
                    self.best_valid_loss[module] = valid_loss
                pbar.set_description(f"Validation Loss: {valid_loss:.3f}")
                model.train()
                
        except KeyboardInterrupt:
            save = input("Training stopped. Save model (y/n)?").strip().lower() == 'y'
            if save:
                save_model()
                print("Model saved.")
    
    def train_autoencoder(self, n_epochs, batch_size, lr):
        self._train_module('autoencoder', n_epochs=n_epochs, 
                           batch_size=batch_size, lr=lr)
    
    def train_flow(self, n_epochs, batch_size, lr):
        if not self.up_to_date_latents:
            self._compute_latents()
        self._train_module('flow', n_epochs=n_epochs,
                           batch_size=batch_size, lr=lr)
                
    def _compute_latents(self):
        self.autoencoder.eval()
        train_loader, valid_loader = self.make_loaders(
            self.data['train_data'], self.data['valid_data'], 64, drop_last=False)
        
        with torch.no_grad():
            z = [self.autoencoder.encode(data[0].to(self.device)).detach().cpu().numpy()
                 for data in train_loader]
            self.data['train_latents'] = np.concatenate(z)
            z = [self.autoencoder.encode(data[0].to(self.device)).detach().cpu().numpy()
                 for data in valid_loader]
            self.data['valid_latents'] = np.concatenate(z)
        
        self.scaler = StandardScaler()
        self.data['norm_train_latents'] = self.scaler.fit_transform(self.data['train_latents'])
        self.data['norm_valid_latents'] = self.scaler.transform(self.data['valid_latents'])
        self.up_to_date_latents = True
        
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
        self.autoencoder.eval()
        x = self.to_tensor(x)
        z = self.autoencoder.encode(x)
        return z
    
    def decode(self, z):
        self.autoencoder.eval()
        z = self.to_tensor(z)
        x = self.autoencoder.decode(z)
        return x
    
    def reconstruct(self, x):
        self.autoencoder.eval()
        x = self.to_tensor(x)
        rx = self.autoencoder.reconstruct(x)
        return rx
    
    def log_prob(self, x):
        self.flow.eval()
        self.autoencoder.eval()
        x = self.to_tensor(x)
        z = self.encode(x)
        log_prob = self.flow.log_prob(z)
        return log_prob
        