import os
import time
import io, json
import h5py
import pickle
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

from ulmo.plotting import load_palette, grid_plot
from ulmo.utils import HDF5Dataset, id_collate, get_quantiles
from ulmo.models import DCAE, ConditionalFlow


try:
    import cartopy.crs as ccrs
except:
    print("Cartopy not installed.  Some plots will not work!")

class ProbabilisticAutoencoder:
    @classmethod
    def from_json(cls, json_file, **kwargs):
        # Load JSON
        with open(json_file, 'rt') as fh:
            model_dict = json.load(fh)
        # Tuples
        tuples = ['image_shape']
        for key in model_dict.keys():
            for sub_key in model_dict[key]:
                if sub_key in tuples:
                    model_dict[key][sub_key] = tuple(model_dict[key][sub_key])
        # Instatiate the pieces
        autoencoder = DCAE(**model_dict['AE'])
        flow = ConditionalFlow(**model_dict['flow'])
        # Do it!
        pae = cls(autoencoder=autoencoder, flow=flow, write_model=False, **kwargs)
        return pae

    """A probabilistic autoencoder (see arxiv.org/abs/2006.05479)."""
    def __init__(self, autoencoder, flow, filepath, datadir=None, 
                 logdir=None, device=None, skip_mkdir=False,
                 write_model=True):
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
            skip_mkdir: bool, optional
                If True, do not make any dirs
        """
        if logdir is None:
            logdir = './'
        if datadir is None:
            datadir = os.path.split(filepath)[0]
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not skip_mkdir:
            if not os.path.isdir(logdir):
                os.makedirs(logdir)
            if not os.path.isdir(datadir):
                os.makedirs(datadir)
        
        self.device = device
        self.scaler = None
        self.autoencoder = autoencoder.to(device)
        self.flow = flow.to(device)
        self.stem = os.path.splitext(os.path.split(filepath)[1])[0]
        self.filepath = {
            'data': filepath,
            'latents': os.path.join(datadir, self.stem + '_latents.h5'),
            'log_probs': os.path.join(datadir, self.stem + '_log_probs.h5'),
            'flow_latents': os.path.join(datadir, self.stem + '_flow_latents.h5')}
        self.savepath = {
            'model': os.path.join(logdir, 'model.json'),
            'flow': os.path.join(logdir, 'flow.pt'),
            'autoencoder': os.path.join(logdir, 'autoencoder.pt')}
        self.logdir = logdir
        self.datadir = datadir
        self.up_to_date_latents = False
        self.up_to_date_log_probs = False
        self.up_to_date_flow_latents = False
        assert flow.dim == autoencoder.latent_dim
        
        self.best_valid_loss = {
            'flow': np.inf,
            'autoencoder': np.inf}

        # Write model to JSON
        if write_model:
            self.write_model()
        
    def save_autoencoder(self):
        torch.save(self.autoencoder.state_dict(), self.savepath['autoencoder'])
        
    def load_autoencoder(self):
        print(f"Loading autoencoder model from: {self.savepath['autoencoder']}")
        self.autoencoder.load_state_dict(torch.load(self.savepath['autoencoder'], map_location=self.device))
        
    def save_flow(self):
        torch.save(self.flow.state_dict(), self.savepath['flow'])
    
    def load_flow(self):
        print(f"Loading flow model from: {self.savepath['flow']}")
        self.flow.load_state_dict(torch.load(self.savepath['flow'], map_location=self.device))

    def write_model(self):
        # Generate the dict
        ood_model = {}
        # AE
        ood_model['AE'] = dict(image_shape=(self.autoencoder.c, self.autoencoder.w, self.autoencoder.h),
                               latent_dim=self.autoencoder.latent_dim)
        # Flow
        flow_attr = ['dim', 'context_dim', 'transform_type', 'n_layers', 'hidden_units', 'n_blocks',
                     'dropout', 'use_batch_norm', 'tails', 'tail_bound', 'n_bins', 'min_bin_height',
                     'min_bin_width', 'min_derivative', 'unconditional_transform', 'encoder']
        ood_model['flow'] = {}
        for attr in flow_attr:
            ood_model['flow'][attr] = getattr(self.flow, attr)
        # JSONify
        with io.open(self.savepath['model'], 'w', encoding='utf-8') as f:
            f.write(json.dumps(ood_model, sort_keys=True, indent=4,
                           separators=(',', ': ')))
        print(f"Wrote model parameters to {self.savepath['model']}")

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
            save = input("Training stopped. Save model (y/n)?").strip().lower() == 'y'
            if save:
                save_model()
                print("Model saved.")

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
        if os.path.isfile(self.filepath['latents']):
            compute = input("Existing latents file found. Use file (y) or recompute (n)?").strip().lower() == 'n'
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
                         for data in tqdm(train_loader, total=len(train_loader), unit='batch', 
                                          desc='Computing train latents')]
                    train = self.scaler.fit_transform(np.concatenate(z))
                    f.create_dataset('train', data=train); del train
                    z = [self.autoencoder.encode(data[0].to(self.device)).detach().cpu().numpy()
                         for data in tqdm(valid_loader, total=len(valid_loader), unit='batch',
                                          desc='Computing valid latents')]
                    valid = self.scaler.transform(np.concatenate(z))
                    f.create_dataset('valid', data=valid); del valid
            
            scaler_path = os.path.join(self.logdir, self.stem + '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        self.up_to_date_latents = True
    
    def _compute_log_probs(self):
        if os.path.isfile(self.filepath['log_probs']):
            compute = input("Existing log probs file found. Use file (y) or recompute (n)?").strip().lower() == 'n'
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
                         for data in tqdm(train_loader, total=len(train_loader), unit='batch',
                                          desc='Computing train log probs')]
                    f.create_dataset('train', data=np.concatenate(log_prob))
                    log_prob = [self.flow.log_prob(data[0].to(self.device)).detach().cpu().numpy()
                         for data in tqdm(valid_loader, total=len(valid_loader), unit='batch',
                                          desc='Computing valid log probs')]
                    f.create_dataset('valid', data=np.concatenate(log_prob))
        self.up_to_date_log_probs = True
        
    def _compute_flow_latents(self):
        if os.path.isfile(self.filepath['flow_latents']):
            compute = input("Existing flow latents file found. Use file (y) or recompute (n)?").strip().lower() == 'n'
        else:
            compute = True
        if compute:
            self.flow.eval()
            if not self.up_to_date_latents:
                self._compute_latents()
            train_loader, valid_loader = self._make_loaders(
                'latents', batch_size=1024, drop_last=False)

            with h5py.File(self.filepath['flow_latents'], 'w') as f:
                with torch.no_grad():
                    log_prob = [self.flow.latent_representation(data[0].to(self.device)).detach().cpu().numpy()
                         for data in tqdm(train_loader, total=len(train_loader), unit='batch',
                                         desc='Computing train flow latents')]
                    f.create_dataset('train', data=np.concatenate(log_prob))
                    log_prob = [self.flow.latent_representation(data[0].to(self.device)).detach().cpu().numpy()
                         for data in tqdm(valid_loader, total=len(valid_loader), unit='batch',
                                         desc='Computing valid flow latents')]
                    f.create_dataset('valid', data=np.concatenate(log_prob))
        self.up_to_date_flow_latents = True

    def load_meta(self, key, filename=None):
        """

        Parameters
        ----------
        key : str
            Group name for metadata, e.g. "valid_metadata"
        filename : str, optional
            File for meta data

        Returns
        -------
        df : pandas.DataFrame

        """
        if filename is None:
            filename = self.filepath['data']
        # Proceed
        with h5py.File(filename, 'r') as f:
            if key in f.keys():
                meta = f[key]
                df = pd.DataFrame(meta[:].astype(np.unicode_), columns=meta.attrs['columns'])
            else:
                df = pd.DataFrame()
        return df

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
        """
        Use the AutoEncoder to reconstruct an input image

        Parameters
        ----------
        x : np.ndarray
            Image

        Returns
        -------
        rx : np.ndarray
            Reconstructed image

        """
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
    
    def compute_log_probs(self, input_file, dataset, output_file,
                          scaler=None, csv=False, query=False):
        """
        Computer log probs on an input HDF file of images

        Parameters
        ----------
        input_file
        dataset
        output_file
        scaler
        query : bool, optional
            If True, query the user

        Returns
        -------

        """
        if scaler is None:
            scaler = self.scaler
            scaler_path = os.path.join(self.logdir, self.stem + '_scaler.pkl')
            if self.scaler is None:
                if os.path.exists(scaler_path):
                    if query:
                        load = input("Scaler file found in logdir. Use this (y/n)?") == 'y'
                    else:
                        load = True
                    if load:
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                    else:
                        raise RuntimeError("No scaler provided. Saved scaler found but not loaded.")
                else:
                    raise RuntimeError("No scaler found or provided.")

        # Make PyTorch dataset from HDF5 file
        assert input_file.endswith('.h5'), "Input file must be in .h5 format."
        assert output_file.endswith('.h5'), "Output file must be in .h5 format."
        
        dset = HDF5Dataset(input_file, partition=dataset)
        loader = torch.utils.data.DataLoader(
            dset, batch_size=1024, shuffle=False, 
            drop_last=False, collate_fn=id_collate,
            num_workers=16)
        
        self.autoencoder.eval()
        self.flow.eval()
        
        with torch.no_grad():
            latents = [self.autoencoder.encode(data[0].to(self.device)).detach().cpu().numpy()
                     for data in loader]

        latents = scaler.transform(np.concatenate(latents))

        dset = torch.utils.data.TensorDataset(torch.from_numpy(latents).float())
        loader = torch.utils.data.DataLoader(
            dset, batch_size=1024, shuffle=False, 
            drop_last=False, num_workers=16)
        
        with h5py.File(output_file, 'w') as f:
            with torch.no_grad():
                log_prob = [self.flow.log_prob(data[0].to(self.device)).detach().cpu().numpy()
                     for data in tqdm(loader, total=len(loader), unit='batch', desc='Computing log probs')]
                f.create_dataset(dataset, data=np.concatenate(log_prob))
        print(f"Log probabilities saved to {output_file}.")

        # CSV?
        if csv:
            csv_name = output_file.replace('h5', 'csv')
            df = self.load_meta(dataset+'_metadata', filename=input_file)
            self._log_probs_to_csv(df, output_file, csv_name,
                                   dataset=dataset)

    def _log_probs_to_csv(self, df, log_file, outfile, dataset='valid'):
        """

        Parameters
        ----------
        df : pnadas.DataFrame
        log_file : str
        outfile : str
        dataset : str, optional

        """
        with h5py.File(log_file, 'r') as f:
            df['log_likelihood'] = f[dataset][:].flatten()
        df.to_csv(outfile, index=False)
        print(f"Saved log probabilities to {outfile}.")

    def save_log_probs(self):
        """
        Write the log probabilities to a CSV file

        Returns
        -------

        """
        if not self.up_to_date_log_probs:
            self._compute_log_probs()

        # Prep
        df = self.load_meta('valid_metadata')
        csv_name = self.stem + '_log_probs.csv'
        outfile = os.path.join(self.logdir, csv_name)

        # Call
        self._log_probs_to_csv(df, self.filepath['log_probs'], outfile)



    def plot_reconstructions(self, save_figure=False, ivmnx=(-2,2)):
        """
        Generate a grid of plots of reconstructed images

        Parameters
        ----------
        save_figure : bool, optional
        ivmnx : tuple, opional

        """
        pal, cm = load_palette()
        
        with h5py.File(self.filepath['data'], 'r') as f:
            fields = f['valid']
            n = fields.shape[0]
            idx = sorted(np.random.choice(n, replace=False, size=16))
            fields = fields[idx]
            recons = self.reconstruct(fields)
            df = self.load_meta('valid_metadata')

        fig, axes = grid_plot(nrows=4, ncols=4)

        for i, (ax, r_ax) in enumerate(axes):
            x = fields[i, 0]
            rx = recons[i, 0]
            ax.axis('equal')
            r_ax.axis('equal')
            if df is not None:
                file, row, col = df.iloc[i][['filename', 'row', 'column']]
                ax.set_title(f'{file}')
                t = ax.text(0.12, 0.89, f'({row}, {col})', color='k', size=12, transform=ax.transAxes)
                t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            else:
                ax.set_title(f'Image {i}\n{logL_title}')
            if ivmnx[0] is None:
                vmnx = np.min(x), np.max(x)
            else:
                vmnx = ivmnx
            sns.heatmap(x, ax=ax, xticklabels=[], yticklabels=[], cmap=cm,
                    vmin=vmnx[0], vmax=vmnx[1])
            sns.heatmap(rx, ax=r_ax, xticklabels=[], yticklabels=[], cmap=cm,
                    vmin=vmnx[0], vmax=vmnx[1])
        if save_figure:
            fig_name = 'grid_reconstructions_' + self.stem + '.png'
            plt.savefig(os.path.join(self.logdir, fig_name), bbox_inches='tight')
        plt.show()
    
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
            fig_name = 'log_probs_' + self.stem + '.png'
            plt.savefig(os.path.join(self.logdir, fig_name), bbox_inches='tight')
        plt.show()

    def plot_grid(self, kind, save_metadata=False, save_figure=False,
                 vmin=None, vmax=None, grad_vmin=None, grad_vmax=None):
        
        vmin = vmin if vmin is not None else -2
        vmax = vmax if vmax is not None else 2
        grad_vmin = grad_vmin if grad_vmin is not None else 0
        grad_vmax = grad_vmax if grad_vmax is not None else 1
        
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
                logL > np.quantile(logL, kind-0.01),
                logL < np.quantile(logL, kind+0.01))
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
                meta = f['valid_metadata']
                df = pd.DataFrame(meta[idx].astype(np.unicode_), columns=meta.attrs['columns'])
                if save_metadata:
                    csv_name = str(kind).replace(' ', '_') + '_metadata.csv'
                    df.to_csv(os.path.join(self.logdir, csv_name), index=False)
            else:
                meta = None
        
        fig, axes = grid_plot(nrows=4, ncols=4)

        for i, (ax, grad_ax) in enumerate(axes):
            field = fields[i, 0]
            grad = filters.sobel(field)
            p = sum(field_logL[i] > logL)
            n = len(logL)
            logL_title = f'Likelihood quantile: {p/n:.5f}'
            ax.axis('equal')
            grad_ax.axis('equal')
            if meta is not None:
                file, row, col = df.iloc[i][['filename', 'row', 'column']]
                ax.set_title(f'{file}\n{logL_title}')
                t = ax.text(0.12, 0.89, f'({row}, {col})', color='k', size=12, transform=ax.transAxes)
                t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            else:
                ax.set_title(f'Image {i}\n{logL_title}')
            sns.heatmap(field, ax=ax, xticklabels=[], yticklabels=[], cmap=cm, vmin=vmin, vmax=vmax)
            sns.heatmap(grad, ax=grad_ax, xticklabels=[], yticklabels=[], cmap=cm, vmin=grad_vmin, vmax=grad_vmax)
        if save_figure:
            fig_name = 'grid_' + str(kind).replace(' ', '_') + '_' + self.stem + '.png'
            plt.savefig(os.path.join(self.logdir, fig_name), bbox_inches='tight')
        plt.show()
        
    def plot_geographical(self, save_figure=False):
        csv_name = self.stem + '_log_probs.csv'
        filepath = os.path.join(self.logdir, csv_name)
        
        if not os.path.isfile(filepath):
            self.save_log_probs()
        
        log_probs = pd.read_csv(filepath)
        log_probs['quantiles'] = get_quantiles(log_probs['log_likelihood'])
        
        fig = plt.figure(figsize=(25, 16))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        plt.scatter(log_probs['longitude'], log_probs['latitude'], 
            c=log_probs['quantiles'], cmap=plt.get_cmap('jet_r'),
            s=10, alpha=0.4)
        cbar = plt.colorbar(fraction=0.0231, pad=0.02)
        cbar.set_label('Likelihood Quantile')
        if save_figure:
            fig_name = 'geo_' + self.stem + '.png'
            plt.savefig(os.path.join(self.logdir, fig_name), bbox_inches='tight')
        plt.show()