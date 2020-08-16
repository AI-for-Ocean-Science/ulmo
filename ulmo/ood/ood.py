import os
import time
import h5py
import sklearn
import multiprocessing
import numpy as np
import pandas as pd
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
from ulmo.utils import HDF5Dataset, id_collate


class ProbabilisticAutoencoder:
    """A probabilistic autoencoder (see arxiv.org/abs/2006.05479)."""
    def __init__(self, autoencoder, flow, filepath, datadir=None, 
                 logdir=None, device=None):
        """
        Parameters
            autoencoder: ulmo.models.Autoencoder
                Autoencoder model for dimensionality reduction
            flow: ulmo.models.ConditionalFlow
                Flow model for likelihood estimation
            filepath: location of .hdf5 file containing training
                and validation data for the OOD task. Should include 
                'train' and 'valid' datasets of shape (*, D) for 
                vector data or (*, C, H, W) for image data and 
                optionally 'train_metadata' and 'valid_metadata' 
                of shape (*, M)
            datadir: str
                Location to store intermediate datasets such
                as latent variables and log probabilities
            logdir: str
                Location to store logs and models
            device: torch.device
                Device to use for training and inference
        """
        if logdir is None:
            logdir = './'
        if datadir is None:
            datadir = os.path.split(filepath)[0]
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.device = device
        self.autoencoder = autoencoder.to(device)
        self.flow = flow.to(device)
        self.stem = os.path.splitext(os.path.split(filepath)[1])[0]
        self.filepath = {
            'data': filepath,
            'latents': os.path.join(datadir, self.stem + '_latents.h5'),
            'log_probs': os.path.join(datadir, self.stem + '_log_probs.h5')}
        self.savepath = {
            'flow': os.path.join(logdir, 'flow.pt'),
            'autoencoder': os.path.join(logdir, 'autoencoder.pt')}
        self.logdir = logdir
        self.datadir = datadir
        self.up_to_date_latents = False
        self.up_to_date_log_probs = False
        assert flow.dim == autoencoder.latent_dim
        
        self.best_valid_loss = {
            'flow': np.inf,
            'autoencoder': np.inf}
        
    def save_autoencoder(self):
        torch.save(self.autoencoder.state_dict(), self.savepath['autoencoder'])
        
    def load_autoencoder(self):
        print(f"Loading autoencoder model from: {self.savepath['autoencoder']}")
        self.autoencoder.load_state_dict(torch.load(self.savepath['autoencoder']))
        
    def save_flow(self):
        torch.save(self.flow.state_dict(), self.savepath['flow'])
    
    def load_flow(self):
        print(f"Loading flow model from: {self.savepath['flow']}")
        self.flow.load_state_dict(torch.load(self.savepath['flow']))
        
    def _make_loaders(self, kind, batch_size, drop_last=True):
        filepath = self.filepath[kind]
            
        train_dset = HDF5Dataset(filepath, partition='train')
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=batch_size, shuffle=False, 
            drop_last=drop_last, collate_fn=id_collate,
            num_workers=16)

        valid_dset = HDF5Dataset(filepath, partition='valid')
        valid_loader = torch.utils.data.DataLoader(
            valid_dset, batch_size=batch_size, shuffle=False, 
            drop_last=drop_last, collate_fn=id_collate,
            num_workers=16)
        
        print(f"{len(train_loader)*train_loader.batch_size:,d} training samples. "
              f"{len(valid_loader)*valid_loader.batch_size:,d} validation samples.")

        return train_loader, valid_loader
        
    def _train_module(self, module, n_epochs, batch_size, lr,
                     summary_interval=50, eval_interval=500,
                     show_plots=True):
        try:
            module = module.strip().lower()
            if module == 'autoencoder':
                model = self.autoencoder
                save_model = self.save_autoencoder
                load_model = self.load_autoencoder
                train_loader, valid_loader = self._make_loaders('data', batch_size)
                self.up_to_date_latents = False
            elif module == 'flow':
                model = self.flow
                save_model = self.save_flow
                load_model = self.load_flow
                train_loader, valid_loader = self._make_loaders('latents', batch_size)
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
        print("Computing latent representations...")
        if os.path.isfile(self.filepath['latents']):
            compute = input("Existing file found. Use file (y) or recompute (n)?").strip().lower() == 'n'
        else:
            compute = True
        if compute:
            self.autoencoder.eval()
            train_loader, valid_loader = self._make_loaders(
                'data', batch_size=2048, drop_last=False)

            self.scaler = StandardScaler()
            with h5py.File(self.filepath['latents'], 'w') as f:
                with torch.no_grad():
                    z = [self.autoencoder.encode(data[0].to(self.device)).detach().cpu().numpy()
                         for data in tqdm(train_loader, total=len(train_loader), unit='batch')]
                    train = self.scaler.fit_transform(np.concatenate(z))
                    f.create_dataset('train', data=train); del train
                    z = [self.autoencoder.encode(data[0].to(self.device)).detach().cpu().numpy()
                         for data in tqdm(valid_loader, total=len(valid_loader), unit='batch')]
                    valid = self.scaler.transform(np.concatenate(z))
                    f.create_dataset('valid', data=valid); del valid
        self.up_to_date_latents = True
    
    def _compute_log_probs(self):
        print("Computing log probabilities...")
        if os.path.isfile(self.filepath['log_probs']):
            compute = input("Existing file found. Use file (y) or recompute (n)?").strip().lower() == 'n'
        else:
            compute = True
        if compute:
            self.flow.eval()
            if not self.up_to_date_latents:
                self._compute_latents()
            train_loader, valid_loader = self._make_loaders(
                'latents', batch_size=1024, drop_last=False)

            with h5py.File(self.filepath['log_probs'], 'w') as f:
                with torch.no_grad():
                    log_prob = [self.flow.log_prob(data[0].to(self.device)).detach().cpu().numpy()
                         for data in tqdm(train_loader, total=len(train_loader))]
                    f.create_dataset('train', data=np.concatenate(log_prob))
                    log_prob = [self.flow.log_prob(data[0].to(self.device)).detach().cpu().numpy()
                         for data in tqdm(valid_loader, total=len(valid_loader))]
                    f.create_dataset('valid', data=np.concatenate(log_prob))
        self.up_to_date_log_probs = True
        
    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
            return x
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
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
    
    def plot_log_probs(self, sample_size=10000, save_figure=False):
        if not self.up_to_date_log_probs:
            self._compute_log_probs()
        
        with h5py.File(self.filepath['log_probs'], 'r') as f:
            logL = f['valid'][:].flatten()
        if len(logL) > sample_size:
            logL = sklearn.utils.shuffle(logL)[:sample_size]
        low_logL = np.quantile(logL, 0.05)
        high_logL = np.quantile(logL, 0.95)
        sns.distplot(logL)
        plt.axvline(low_logL, linestyle='--', c='r')
        plt.axvline(high_logL, linestyle='--', c='r')
        plt.xlabel('Log Likelihood')
        plt.ylabel('Probability Density')
        if save_figure:
            plt.savefig(os.path.join(self.logdir, 'log_probs'))
        plt.show()

    def plot_grid(self, kind, save_figure=False):
        if not self.up_to_date_log_probs:
            self._compute_log_probs()
        
        pal, cm = load_palette()
        
        with h5py.File(self.filepath['log_probs'], 'r') as f:
            logL = f['valid'][:].flatten()
        if kind == 'outliers':
            mask = logL < np.quantile(logL, 0.05)
        elif kind == 'inliers':
            mask = logL > np.quantile(logL, 0.95)
        elif kind == 'midliers':
            mask = np.logical_and(
                logL > np.quantile(logL, 0.4),
                logL < np.quantile(logL, 0.6))
        elif kind == 'most likely':
            indices = np.argsort(logL)[::-1][:16]
            mask = np.array([False] * len(logL))
            mask[indices] = True
        elif kind == 'least likely':
            indices = np.argsort(logL)[:16]
            mask = np.array([False] * len(logL))
            mask[indices] = True
        elif (isinstance(kind, (int, float))
              and not isinstance(kind, bool)):
            mask = np.logical_and(
                logL > np.quantile(logL, kind-0.05),
                logL < np.quantile(logL, kind+0.05))
        else:
            raise ValueError(f"Kind {kind} unknown.")
        
        # Indices of log probs should align with fields since we
        # are *not* shuffling when creating data loaders.
        
        # Select 16 random indices from mask
        idx = sorted(np.random.choice(np.where(mask)[0], replace=False, size=16))
        field_logL = logL[idx]
        
        # Retrieve data
        with h5py.File(self.filepath['data'], 'r') as f:
            fields = f['valid'][idx]
            if 'valid_metadata' in f.keys():
                meta = f['valid_metadata'][idx].astype(np.unicode_)
                # Output metadata to file
                meta_df = pd.DataFrame(meta, 
                    columns=['filename', 'row', 'column', 'mean_temperature', 'clear_fraction'])
                csv_name = str(kind).replace(' ', '_') + '_metadata.csv'
                meta_df.to_csv(os.path.join(self.logdir, csv_name), index=False)
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
            logL_title = f'Likelihood quantile: {p/n:.5f}'
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
        if save_figure:
            fig_name = 'grid_' + str(kind).replace(' ', '_') + '.png'
            plt.savefig(os.path.join(self.logdir, fig_name), bbox_inches='tight')
        plt.show()