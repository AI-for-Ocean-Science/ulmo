import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import ulmo.nflows.nn as nn_
import ulmo.nflows.utils as utils
from ulmo.nflows import distributions, flows, transforms


class ConditionalFlow(nn.Module):
    """A conditional rational quadratic neural spline flow."""
    def __init__(self, dim, context_dim, transform_type, n_layers, hidden_units,
        n_blocks, dropout, use_batch_norm, tails, tail_bound, n_bins,
        min_bin_height, min_bin_width, min_derivative, unconditional_transform,
        encoder=None):
        super().__init__()

        self.dim = dim
        self.context_dim = context_dim
        self.transform_type = transform_type
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.tails = tails
        self.tail_bound = tail_bound
        self.n_bins = n_bins
        self.min_bin_height = min_bin_height
        self.min_bin_width = min_bin_width
        self.min_derivative = min_derivative
        self.unconditional_transform = unconditional_transform
        self.encoder = encoder

        distribution = distributions.StandardNormal([dim])
        transform = transforms.CompositeTransform([
            self.create_transform(self.transform_type)
            for _ in range(self.n_layers)])
        self.flow = flows.Flow(transform, distribution)

    def create_transform(self, type):
        """Create invertible rational quadratic transformations."""
        linear = transforms.RandomPermutation(features=self.dim)
        if type == 'coupling':
            base = transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=utils.create_mid_split_binary_mask(features=self.dim),
                transform_net_create_fn=lambda in_features, out_features:
                    nn_.nets.ResidualNet(
                        in_features=in_features,
                        out_features=out_features,
                        context_features=self.context_dim,
                        hidden_features=self.hidden_units,
                        num_blocks=self.n_blocks,
                        dropout_probability=self.dropout,
                        use_batch_norm=self.use_batch_norm,
                    ),
                tails=self.tails,
                tail_bound=self.tail_bound,
                num_bins=self.n_bins,
                min_bin_height=self.min_bin_height,
                min_bin_width=self.min_bin_width,
                min_derivative=self.min_derivative,
                apply_unconditional_transform=self.unconditional_transform,
            )
        elif type == 'autoregressive':
            base = transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self.dim,
                hidden_features=self.hidden_units,
                context_features=self.context_dim,
                num_bins=self.n_bins,
                tails=self.tails,
                tail_bound=self.tail_bound,
                num_blocks=self.n_blocks,
                use_residual_blocks=True,
                random_mask=False,
                activation=F.relu,
                dropout_probability=self.dropout,
                use_batch_norm=self.use_batch_norm,
        )
        else:
            raise ValueError(f'Transform type {self.transform_type} unavailable.')
        t = transforms.CompositeTransform([linear, base])
        return t

    def log_prob(self, inputs, context=None):
        """Forward pass in density estimation direction.

        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context.
        Returns:
            log_prob (torch.Tensor): [N,] tensor of log probabilities.
        """
        if self.encoder is not None and context is not None:
            context = self.encoder(context)
        log_prob = self.flow.log_prob(inputs, context)
        return log_prob

    def forward(self, inputs, context=None):
        """Forward pass to negative log likelihood (NLL).

        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context.
        Returns:
            loss (torch.Tensor): [1,] tensor of mean NLL
        """
        log_prob = self.log_prob(inputs, context)
        loss = -torch.mean(log_prob)
        return loss

    def sample(self, n_samples, context=None):
        """Draw samples from the conditional flow.

        Args:
            n_samples (int): Number of samples to draw.
            context (torch.Tensor): [context_dim] tensor of conditioning info.
        Returns:
            samples (torch.Tensor): [n_samples, dim] tensor of data
            log_prob (torch.Tensor): [n_samples,] tensor of log probabilities
        """
        if context is not None:
            context = context.unsqueeze(0)
            if self.encoder is not None:
                context = self.encoder(context)
            context = context.expand(n_samples, -1)
            noise = self.flow._distribution.sample(1, context).squeeze(1)
        else:
            noise = self.flow._distribution.sample(n_samples)
        samples, log_prob = self.flow._transform.inverse(noise, context)
        return samples, log_prob
    
    def latent_representation(self, inputs, context=None):
        """Get representations of data in latent space.
        
        Args:
            inputs (torch.Tensor): [*, dim] tensor of data.
            context (torch.Tensor): [*, context_dim] tensor of context.
        Returns:
            latents (torch.Tensor): [*, dim] tensor of latents."""
        latents = self.flow.transform_to_noise(inputs, context)
        return latents
        