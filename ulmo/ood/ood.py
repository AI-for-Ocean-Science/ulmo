import os
import time
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from skimage import filters
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from ulmo.plotting import load_palette



class ProbabilisticAutoencoder:
    """A probabilistic autoencoder (see arxiv.org/abs/2006.05479)."""
    def __init__(self, autoencoder, flow, data, metadata=None,
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
            metadata: np.ndarray
                Array of metadata corresponding to data,
                should be of shape (*, M)
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
        self.up_to_date_log_probs = False
        assert flow.dim == autoencoder.latent_dim
        
        n = data.shape[0]
        idx = sklearn.utils.shuffle(np.arange(n))
        m = int(self.valid_fraction * n)
        valid_idx = idx[:m]
        train_idx = idx[m:]
        
        self.data = {
            'train_data': data[train_idx],
            'valid_data': data[valid_idx],
            'train_metadata': None,
            'valid_metadata': None}
        self.best_valid_loss = {
            'flow': np.inf,
            'autoencoder': np.inf}
        
        if metadata is not None:
            self.data['train_metadata'] = metadata[train_idx]
            self.data['valid_metadata'] = metadata[valid_idx]
        
    def make_loaders(self, train_data, valid_data, batch_size, drop_last=True):
        train_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_data).float())
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=batch_size, shuffle=False, drop_last=drop_last)

        valid_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(valid_data).float())
        valid_loader = torch.utils.data.DataLoader(
            valid_dset, batch_size=batch_size, shuffle=False, drop_last=drop_last)

        return train_loader, valid_loader
    
    def save_autoencoder(self):
        torch.save(self.autoencoder.state_dict(), self.autoencoder_filepath)
        
    def load_autoencoder(self):
        self.autoencoder.load_state_dict(torch.load(self.autoencoder_filepath))
        
    def save_flow(self):
        torch.save(self.flow.state_dict(), self.flow_filepath)
    
    def load_flow(self):
        self.flow.load_state_dict(torch.load(self.flow_filepath))
        
    def _train_module(self, module, n_epochs, batch_size, lr,
                     summary_interval=50, eval_interval=500,
                     show_plots=True):
        try:
            module = module.strip().lower()
            if module == 'autoencoder':
                model = self.autoencoder
                save_model = self.save_autoencoder
                load_model = self.load_autoencoder
                train_loader, valid_loader = self.make_loaders(
                    self.data['train_data'], self.data['valid_data'], batch_size)
                print(f"Training on {self.data['train_data'].shape[0]:,d} samples. "
                      f"Validating on {self.data['valid_data'].shape[0]:,d} samples.")
                self.up_to_date_latents = False
            elif module == 'flow':
                model = self.flow
                save_model = self.save_flow
                load_model = self.load_flow
                train_loader, valid_loader = self.make_loaders(
                    self.data['norm_train_latents'], self.data['norm_valid_latents'], batch_size)
                print(f"Training on {self.data['norm_train_latents'].shape[0]:,d} samples. "
                      f"Validating on {self.data['norm_valid_latents'].shape[0]:,d} samples.")
                self.up_to_date_log_probs = False
            else:
                raise ValueError(f"Module {module} unknown.")
            
            model.train()
            global_step = 0
            total_loss = 0
            train_losses, valid_losses = [], []
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            
            # Train
            epoch_pbar = tqdm(range(n_epochs), unit='epoch')
            for epoch in epoch_pbar:
                batch_pbar = tqdm(train_loader, unit='batch')
                for data in batch_pbar:
                    optimizer.zero_grad()
                    loss = model(data[0].to(self.device))
                    loss.backward()
                    total_loss += loss.item()
                    optimizer.step()
                    global_step += 1
                    
                    if global_step % summary_interval == 0:
                        # Summary
                        mean_loss = total_loss / summary_interval
                        train_losses.append((global_step, mean_loss))
                        total_loss = 0
                        batch_pbar.set_description(f"Training Loss: {mean_loss:.3f}")
                        
                    if global_step % eval_interval == 0:
                        # Evaluate
                        model.eval()
                        with torch.no_grad():
                            total_valid_loss = 0
                            for i, data in enumerate(valid_loader):
                                loss = model(data[0].to(self.device))
                                total_valid_loss += loss.item()
                        valid_loss = total_valid_loss / float(i+1)
                        if valid_loss < self.best_valid_loss[module]:
                            save_model()
                            self.best_valid_loss[module] = valid_loss
                        epoch_pbar.set_description(f"Validation Loss: {valid_loss:.3f}")
                        valid_losses.append((global_step, valid_loss))
                        model.train()
                
        except KeyboardInterrupt:
            save = input("Training stopped. Save model (y/n)?").strip().lower() == 'y'
            if save:
                save_model()
                print("Model saved.")
        
        finally:
            load_model()
            if show_plots:
                train_losses = np.array(train_losses)
                valid_losses = np.array(valid_losses)
                plt.plot(*train_losses.T, label='Training')
                plt.plot(*valid_losses.T, label='Validation')
                plt.xlabel('Global Step')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
    
    def train_autoencoder(self, **kwargs):
        self._train_module('autoencoder', **kwargs)
    
    def train_flow(self, **kwargs):
        if not self.up_to_date_latents:
            self._compute_latents()
        self._train_module('flow', **kwargs)
                
    def _compute_latents(self):
        self.autoencoder.eval()
        train_loader, valid_loader = self.make_loaders(
            self.data['train_data'], self.data['valid_data'], 
            batch_size=256, drop_last=False)
        
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
    
    def _compute_log_probs(self):
        self.flow.eval()
        if not self.up_to_date_latents:
            self._compute_latents()

        train_loader, valid_loader = self.make_loaders(
            self.data['norm_train_latents'], self.data['norm_valid_latents'],
            batch_size=100, drop_last=False)
        
        with torch.no_grad():
            log_prob = [self.flow.log_prob(data[0].to(self.device)).detach().cpu().numpy()
                 for data in train_loader]
            self.data['train_log_probs'] = np.concatenate(log_prob)
            log_prob = [self.flow.log_prob(data[0].to(self.device)).detach().cpu().numpy()
                 for data in valid_loader]
            self.data['valid_log_probs'] = np.concatenate(log_prob)
        self.up_to_date_log_probs = True
        
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
        
    def to_type(self, x, t):
        if t == np.ndarray:
            x = self.to_array(x)
        elif t == torch.Tensor:
            x = self.to_tensor(x)
        else:
            raise ValueError(f"Type {t} not supported.")
        return x
        
    def encode(self, x):
        t = type(x)
        self.autoencoder.eval()
        x = self.to_tensor(x)
        z = self.autoencoder.encode(x)
        z = self.to_type(z, t)
        return z
    
    def decode(self, z):
        t = type(z)
        self.autoencoder.eval()
        z = self.to_tensor(z)
        x = self.autoencoder.decode(z)
        x = self.to_type(x, t)
        return x
    
    def reconstruct(self, x):
        t = type(x)
        self.autoencoder.eval()
        x = self.to_tensor(x)
        rx = self.autoencoder.reconstruct(x)
        rx = self.to_type(rx, t)
        return rx
    
    def log_prob(self, x):
        t = type(x)
        self.flow.eval()
        self.autoencoder.eval()
        x = self.to_tensor(x)
        z = self.encode(x)
        log_prob = self.flow.log_prob(z)
        log_prob = self.to_type(log_prob, t)
        return log_prob
    
    def plot_log_probs(self):
        if not self.up_to_date_log_probs:
            self._compute_log_probs()
        
        logL = self.data['valid_log_probs'].flatten()
        low_logL = np.quantile(logL, 0.05)
        high_logL = np.quantile(logL, 0.95)
        sns.distplot(logL)
        plt.axvline(low_logL, linestyle='--', c='r')
        plt.axvline(high_logL, linestyle='--', c='r')
        plt.xlabel('Log Likelihood')
        plt.ylabel('Probability Density')
        plt.show()

    def plot_random(self, kind):
        if not self.up_to_date_log_probs:
            self._compute_log_probs()
        
        pal, cm = load_palette()
        
        logL = self.data['valid_log_probs'].flatten()
        if kind == 'outliers':
            mask = logL < np.quantile(logL, 0.05)
        elif kind == 'inliers':
            mask = logL > np.quantile(logL, 0.95)
        elif kind == 'midliers':
            mask = np.logical_and(
                logL > np.quantile(logL, 0.4),
                logL < np.quantile(logL, 0.6))
        else:
            raise ValueError(f"Kind {kind} unknown.")
        
        # Indices of log probs should align with fields since we
        # are *not* shuffling when creating data loaders.
        
        fields = self.data['valid_data'][mask]
        field_logL = logL[mask]
        idx = sklearn.utils.shuffle(np.arange(fields.shape[0]))[:16]
        fields = fields[idx]
        field_logL = field_logL[idx]
        
        if self.data['valid_metadata'] is not None:
            meta = self.data['valid_metadata'][mask][idx]
        else:
            meta = None
        
        # Make plot grid
        n, m = 4, 4 # rows, columns
        t, b = 0.9, 0.1 # 1-top space, bottom space
        msp, sp = 0.1, 0.5 # minor spacing, major spacing

        offs = (1+msp)*(t-b)/(2*n+n*msp+(n-1)*sp) # grid offset
        hspace = sp+msp+1 #height space per grid

        gso = GridSpec(n, m, bottom=b+offs, top=t, hspace=hspace)
        gse = GridSpec(n, m, bottom=b, top=t-offs, hspace=hspace)

        fig = plt.figure(figsize=(18, 25))
        axes = []
        for i in range(n*m):
            axes.append((fig.add_subplot(gso[i]), fig.add_subplot(gse[i])))

        for i, (ax, grad_ax) in enumerate(axes):
            field = fields[i, 0]
            grad = filters.sobel(field)
            p = sum(field_logL[i] > logL)
            n = len(logL)
            logL_title = f'Likelihood quantile: {p/n:.3f}'
            ax.axis('equal')
            grad_ax.axis('equal')
            if meta is not None:
                file, row, col = meta[i][:3]
                ax.set_title(f'{file}\n{logL_title}')
                t = ax.text(0.12, 0.89, f'({row}, {col})', color='k', size=12, transform=ax.transAxes)
                t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            else:
                ax.set_title(f'Image {i}\n{logL_title}')
            sns.heatmap(field, ax=ax, xticklabels=[], yticklabels=[], cmap=cm, vmin=-2, vmax=2)
            sns.heatmap(grad, ax=grad_ax, xticklabels=[], yticklabels=[], cmap=cm, vmin=0, vmax=1)
        plt.show()