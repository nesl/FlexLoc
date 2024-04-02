import numpy as np
import torch
import torch.nn as nn
from collections import deque
from einops import rearrange
# They have affine transformation over EVERY SINGLE element, this will lead to many parameters...
# How to use: specify mean_dims tches height width channels' to layernorm over the last three pass 'height width channels'
class ConditionalLN(nn.Module):
    param_queue = deque()
    rescale = True
    # Dim can be numpy array, torch.Size, or python list, of sizes

    # pass dim as (C, H, W)
    def __init__(self, dim, eps=1e-5):
        super(ConditionalLN, self).__init__()
        self.gamma = 1
        self.beta = 0
        self.eps = eps
        if isinstance(dim, int):
            self.num_dims = 1
        else:
            self.num_dims = len(dim)
        self.dim_sizes = dim
    
    # We pop a (batch_size * number_layers) containing tuples of delta gamma and beta off queue
    def forward(self, input):
        if (ConditionalLN.rescale):
            # Batch_Size * 2
            popped_params = ConditionalLN.param_queue.popleft()
            delta_gamma = popped_params[..., 0]
            delta_beta = popped_params[..., 1]
            # Add my 1 and 0 offsets
            new_gamma = self.gamma + delta_gamma
            new_beta = self.beta + delta_beta

            mean_dims = np.arange(len(input.shape) - self.num_dims, len(input.shape))
            mean = torch.mean(input, dim=tuple(mean_dims), keepdims=True)
            variance = torch.var(input, dim=tuple(mean_dims), keepdims=True, unbiased=False)
            normalized_input = (input - mean) / torch.sqrt(variance + self.eps)
            
            if (isinstance(delta_gamma, torch.Tensor)):
                newpattern = "b -> b 1 1 1" if self.num_dims==3 else "b -> b 1 1"
                new_gamma = rearrange(new_gamma, newpattern)
                new_beta = rearrange(new_beta, newpattern)
            return torch.mul(normalized_input, new_gamma) + new_beta
        else:
            mean_dims = np.arange(len(input.shape) - self.num_dims, len(input.shape))
            mean = torch.mean(input, dim=tuple(mean_dims), keepdims=True)
            variance = torch.var(input, dim=tuple(mean_dims), keepdims=True, unbiased=False)
            normalized_input = (input - mean) / torch.sqrt(variance + self.eps)
            return normalized_input