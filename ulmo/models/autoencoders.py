import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from abc import ABC, abstractmethod


def divisible_by_two(k, lower_bound=None):
    if lower_bound is None:
        lower_bound = 0
    i = 0
    while k % 2 == 0 and k > lower_bound:
        k /= 2
        i += 1
    return i, int(k)


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
    """A deep convolutional autoencoder

        Parameters
            image_shape: tuple
                colors, width, height

    """
    
    def __init__(self, image_shape, latent_dim):
        super().__init__()

        self.c, self.w, self.h = image_shape
        assert self.w == self.h, "Image must be square"
        self.n_layers, self.w_ = divisible_by_two(self.w, 4)
        assert self.n_layers >= 1, "Image size not divisible by two"
        self.latent_dim = latent_dim
        
        encoder_layers = [self._encoder_layer(self.c, 32)]
        for i in range(self.n_layers-1):
            in_channels, out_channels = 32*2**i, 32*2**(i+1)
            encoder_layers.append(self._encoder_layer(in_channels, out_channels))
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.mid_channels = out_channels
        self.to_z = nn.Linear(int(self.mid_channels*self.w_**2), latent_dim)
        self.from_z = nn.Linear(latent_dim, int(self.mid_channels*self.w_**2))
        
        decoder_layers = []
        for i in range(self.n_layers-1):
            in_channels, out_channels = self.mid_channels//(2**i), self.mid_channels//(2**(i+1))
            decoder_layers.append(self._decoder_layer(in_channels, out_channels))
        decoder_layers.append(self._decoder_layer(out_channels, out_channels))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.output = nn.Conv2d(out_channels, self.c, 3, 1, 1)

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
        x = self.from_z(z).view(z.size(0), self.mid_channels, self.w_, self.w_)
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
    def from_file(f, cpu=False, **kwargs):
        model = DCAE(**kwargs)
        if cpu:
            model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(f))
        model.eval()
        return model