import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from abc import ABC, abstractmethod


class Autoencoder(ABC):
    """
    An abstract class for implementing priors.
    """
    @abstractmethod
    def encode(self, x):
        """
        Encode data.
        
        Parameters
            x: torch.tensor
                (*, D) or (*, C, H, W) batch of data
        Returns
            z: torch.tensor
                (*, latent_dim) batch of latents
        """
        pass
    
    @abstractmethod
    def decode(self, z):
        """
        Decode latents.
        
        Parameters
            z: torch.tensor
                (*, latent_dim) batch of latents
        Returns
            x: torch.tensor
                (*, D) or (*, C, H, W) batch of data

        """
        pass
    
    @abstractmethod
    def reconstruct(self, x):
        """
        Compute reconstruction of x.
        
        Parameters
            x: torch.tensor
                (*, D) or (*, C, H, W) batch of data
        Returns
            rx: torch.tensor
                (*, D) or (*, C, H, W) batch of reconstructions
        """
        pass

    
class DCAE(Autoencoder, nn.Module):
    """A deep convolutional autoencoder."""
    
    def __init__(self, input_channels, latent_dim):
        super().__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            self._encoder_layer(input_channels, 32),
            self._encoder_layer(32, 64),
            self._encoder_layer(64, 128),
            self._encoder_layer(128, 256))
        
        self.to_z = nn.Linear(256*3*3, latent_dim)
        self.from_z = nn.Linear(latent_dim, 256*3*3)
        
        self.decoder = nn.Sequential(
            self._decoder_layer(256, 128),
            self._decoder_layer(128, 64),
            self._decoder_layer(64, 32),
            self._decoder_layer(32, 32))
        
        self.output = nn.Conv2d(32, 1, 3, 1, 1)

    def _encoder_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels,
                kernel_size=3, 
                stride=2, 
                padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
        return layer
    
    def _decoder_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU())
        return layer
    
    def encode(self, x):
        x = self.encoder(x)
        z = self.to_z(x.view(x.size(0), -1))
        return z
    
    def decode(self, z):
        x = self.from_z(z).view(z.size(0), 256, 3, 3)
        x = self.output(self.decoder(x))
        return x
    
    def reconstruct(self, x):
        z = self.encode(x)
        rx = self.decode(z)
        return rx
    
    def forward(self, x):
        rx = self.reconstruct(x)
        return F.mse_loss(x, rx)
    
    @staticmethod
    def from_file(f, **kwargs):
        model = DCAE(**kwargs)
        model.load_state_dict(torch.load(f))
        model.eval()
        return model