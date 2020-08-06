import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from abc import ABC, abstractmethod


class Autoencoder(ABC):
    """
    An abstract class for implementing priors.
    """
    @property
    @abstractmethod
    def latent_dim(self):
        pass
    
    @abstractmethod
    def encode(self, x):
        """
        Encode data.
        
        Parameters
            x: torch.tensor or np.ndarray
                (*, D) or (*, C, H, W) batch of data
        Returns
            z: torch.tensor or np.ndarray
                (*, latent_dim) batch of latents
        """
        pass
    
    @abstractmethod
    def decode(self, z):
        """
        Decode latents.
        
        Parameters
            z: torch.tensor or np.ndarray
                (*, latent_dim) batch of latents
        Returns
            x: torch.tensor or np.ndarray
                (*, D) or (*, C, H, W) batch of data

        """
        pass
    
    @abstractmethod
    def reconstruct(self, x):
        """
        Compute reconstruction of x.
        
        Parameters
            x: torch.tensor or np.ndarray
                (*, D) or (*, C, H, W) batch of data
        Returns
            rx: torch.tensor or np.ndarray
                (*, D) or (*, C, H, W) batch of reconstructions
        """
        pass