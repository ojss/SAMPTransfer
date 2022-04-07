import torch
import torch.nn as nn


class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=.999, T=.07, mlp=False):
        """
           dim: feature dimension (default: 128)
           K: queue size; number of negative keys (default: 65536)
           m: moco momentum of updating key encoder (default: 0.999)
           T: softmax temperature (default: 0.07)
       """
        super(MoCo, self).__init__()
        self.K = K
        self.dim = dim
        self.m = m
        self.T = T

        
