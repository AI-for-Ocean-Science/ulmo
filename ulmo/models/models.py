import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import nsflow.nn as nn_
import nsflow.utils as utils
from .partial_conv import PartialConv2d
from nsflow.nde import distributions, flows, transforms


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, 
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out
    

class PartialConvBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = PartialConv2d(
            inplanes, planes, kernel_size=1, bias=False, 
            multi_channel=True, return_mask=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = PartialConv2d(
            planes, planes, kernel_size=3, stride=stride, 
            padding=1, bias=False, multi_channel=True, return_mask=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = PartialConv2d(
            planes, inplanes, kernel_size=1, bias=False, 
            multi_channel=True, return_mask=True)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, mask):
        residual = x

        out, mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, mask = self.conv2(out, mask)
        out = self.bn2(out)
        out = self.relu(out)

        out, mask = self.conv3(out, mask)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out, mask
    

# class EncoderLayer(nn.Module):
#     def __init__(self, in_features, out_features, n_blocks):
#         super().__init__()
#         self.conv = nn.Conv2d(
#             in_features, out_features, kernel_size=3, 
#             stride=2, padding=1, bias=False)
#         self.act = nn.ReLU()
#         self.bn = nn.BatchNorm2d(out_features)
#         self.blocks = nn.ModuleList([
#             Bottleneck(out_features, out_features // 4) 
#             for _ in range(n_blocks)])
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(self.act(x))
#         for b in self.blocks:
#             x = b(x)
#         return x


class EncoderLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 n_blocks, partial_conv=False):
        super().__init__()
        
        if partial_conv:
            self.conv = PartialConv2d(
                in_features, out_features, kernel_size=3, 
                stride=2, padding=1, bias=False, 
                multi_channel=True, return_mask=True)
            self.blocks = nn.ModuleList([
                PartialConvBottleneck(out_features, out_features // 4) 
                for _ in range(n_blocks)])
        else:
            self.conv = nn.Conv2d(
                in_features, out_features, kernel_size=3, 
                stride=2, padding=1, bias=False)
            self.blocks = nn.ModuleList([
                Bottleneck(out_features, out_features // 4) 
                for _ in range(n_blocks)])
            
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_features)
        self.partial_conv = partial_conv
    
    def forward(self, x, mask=None):
        if self.partial_conv:
            x, mask = self.conv(x, mask)
            x = self.bn(self.act(x))
            for b in self.blocks:
                x, mask = b(x, mask)
            return x, mask
        else:
            x = self.conv(x)
            x = self.bn(self.act(x))
            for b in self.blocks:
                x = b(x)
            return x


class DecoderLayer(nn.Module):
    def __init__(self, in_features, out_features,
                n_blocks, scale_factor=2, pad=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.pad = pad
        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size=3, 
            stride=1, padding=1, bias=False)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_features)
        self.blocks = nn.ModuleList([
            Bottleneck(in_features, in_features * 4) 
            for _ in range(n_blocks)])
    
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        x = F.interpolate(x, scale_factor=self.scale_factor)
        idx = self.pad * self.scale_factor
        x = x[:, :, idx:-idx, idx:-idx]
        x = self.conv(x)
        x = self.bn(self.act(x))
        return x

    
class AutoEncoder(nn.Module):
    
    @staticmethod
    def from_file(f, **kwargs):
        model = AutoEncoder(**kwargs)
        model.load_state_dict(torch.load(f))
        model.eval()
        return model
    
    def __init__(self, latent_dim=256, blocks_per_layer=0):
        super().__init__()
    
        self.encoder = nn.ModuleList([
            EncoderLayer(1, 32, blocks_per_layer),
            EncoderLayer(32, 64, blocks_per_layer),
            EncoderLayer(64, 128, blocks_per_layer),
            EncoderLayer(128, 256, blocks_per_layer),
            EncoderLayer(256, 512, blocks_per_layer)])
        self.e_to_z = nn.Linear(3*3*512, latent_dim)
        self.z_to_d = nn.Linear(latent_dim, 3*3*512)
        self.decoder = nn.ModuleList([
            DecoderLayer(512, 256, blocks_per_layer),
            DecoderLayer(256, 128, blocks_per_layer),
            DecoderLayer(128, 64, blocks_per_layer),
            DecoderLayer(64, 32, blocks_per_layer),
            DecoderLayer(32, 32, blocks_per_layer)])
        self.out = nn.Conv2d(32, 1, 1, 1)
    
    def _forward(self, x):
        bsz = x.size(0)
        for layer in self.encoder:
            x = layer(x)
        z = self.e_to_z(x.view(bsz, -1))
        x = self.z_to_d(z).view(bsz, 512, 3, 3)
        for layer in self.decoder:
            x = layer(x)
        x = self.out(x)
        return x, z
    
    def forward(self, x):
        rx, _ = self._forward(x)
        loss = F.mse_loss(x, rx)
        return loss
    

class PartialConvAutoEncoder(nn.Module):
    
    @staticmethod
    def from_file(f, **kwargs):
        model = PartialConvAutoEncoder(**kwargs)
        model.load_state_dict(torch.load(f))
        model.eval()
        return model
    
    def __init__(self, latent_dim=256, blocks_per_layer=0):
        super().__init__()
    
        self.encoder = nn.ModuleList([
            EncoderLayer(1, 32, blocks_per_layer, partial_conv=True),
            EncoderLayer(32, 64, blocks_per_layer, partial_conv=True),
            EncoderLayer(64, 128, blocks_per_layer, partial_conv=True),
            EncoderLayer(128, 256, blocks_per_layer, partial_conv=True),
            EncoderLayer(256, 512, blocks_per_layer, partial_conv=True)])
        self.e_to_z = nn.Linear(3*3*512, latent_dim)
        self.z_to_d = nn.Linear(latent_dim, 3*3*512)
        self.decoder = nn.ModuleList([
            DecoderLayer(512, 256, blocks_per_layer),
            DecoderLayer(256, 128, blocks_per_layer),
            DecoderLayer(128, 64, blocks_per_layer),
            DecoderLayer(64, 32, blocks_per_layer),
            DecoderLayer(32, 32, blocks_per_layer)])
        self.out = nn.Conv2d(32, 1, 1, 1)
    
    def _forward(self, x, mask):
        bsz = x.size(0)
        for layer in self.encoder:
            x, mask = layer(x, mask)
        assert mask.sum().item() == mask.nelement()
        z = self.e_to_z(x.view(bsz, -1))
        x = self.z_to_d(z).view(bsz, 512, 3, 3)
        for layer in self.decoder:
            x = layer(x)
        x = self.out(x)
        return x, z
    
    def forward(self, x, mask):
        rx, _ = self._forward(x, mask)
        loss = F.mse_loss(x, rx)
        return loss


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
                    nn_.ResidualNet(
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
            context (torch.Tensor): [N, context_dim] tensor of context."""
        if self.encoder is not None and context is not None:
            context = self.encoder(context)
        log_prob = self.flow.log_prob(inputs, context)
        return log_prob

    def forward(self, inputs, context=None):
        """Forward pass to negative log likelihood (NLL).

        Args:
            inputs (torch.Tensor): [N, dim] tensor of data.
            context (torch.Tensor): [N, context_dim] tensor of context."""
        log_prob = self.log_prob(inputs, context)
        loss = -torch.mean(log_prob)
        return loss

    def sample(self, n_samples, context=None):
        """Draw samples from the conditional flow.

        Args:
            n_samples (int): Number of samples to draw.
            context (torch.Tensor): [context_dim] tensor of conditioning info."""
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