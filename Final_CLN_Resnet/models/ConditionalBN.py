import numpy as np
import torch
import torch.nn as nn
from collections import deque
# They have affine transformation over EVERY SINGLE element, this will lead to many parameters...
# How to use: specify the dimensions that you want to layernorm over
# Ex, with 'batch patches height width channels' to layernorm over the last three pass 'height width channels'
class ConditionalBN(nn.Module):
    
    param_queue = deque()
    # Dim can be numpy array, torch.Size, or python list, of sizes
    def __init__(self, dim, param_queue, eps=1e-5):
        super(ConditionalBN, self).__init__()
        self.gamma = 1
        self.beta = 0
        self.eps = eps
        if isinstance(dim, int):
            self.num_dims = 1
        else:
            self.num_dims = len(dim)
        self.dim_sizes = dim
        self.param_queue = param_queue
    # Batch Norm input (Batch_Size, C, H, W), we have C gammas and betas
    def forward(self, input):
        # C gammas and betas
        delta_gamma, delta_beta = ConditionalBN.param_queue.pop()
        new_gamma = self.gamma + delta_gamma
        new_beta = self.beta + delta_beta

        mean = torch.mean(input, dim=(0, 2, 3), keepdims=True)
        variance = torch.var(input, dim=(0, 2, 3), keepdims=True, unbiased=False)
        normalized_input = (input - mean) / torch.sqrt(variance + self.eps)
        # We want to match shape ()
        if (isinstance(delta_gamma, torch.Tensor)):
            new_gamma = torch.unsqueeze(torch.unsqueeze(new_gamma, 1), 2)
            new_beta = torch.unsqueeze(torch.unsqueeze(new_beta, 1), 2)
       
        return torch.mul(normalized_input, new_gamma) + new_beta