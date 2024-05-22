import re

import torch
from torch import nn
from transformers.activations import ACT2FN


class XDPNetProjector(nn.Module):
    def __init__(self, config=None, factor=4):
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.shrink_factor = factor
        self.mlp = nn.Sequential(nn.Linear(inc * self.shrink_factor, ouc), nn.Mish(), nn.Linear(ouc, ouc))

    def forward(self, x):
        num_batches, num_tokens, hidden_size = x.shape
        x = x.view(num_batches, num_tokens // self.shrink_factor, hidden_size * self.shrink_factor)
        x = self.mlp(x)
        return x
    

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)
    elif projector_type.startswith('xdpnet'):
        xdp_factor_match = re.match(r'^xdpnetv10_(\d+)$', projector_type)
        if xdp_factor_match:
            factor = int(xdp_factor_match.group(1))
            return XDPNetProjector(config, factor=factor)
        else:
            return XDPNetProjector(config)

    raise ValueError(f'Unknown projector type: {projector_type}')

