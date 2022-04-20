from typing import Callable, Sequence

import torch
from torch import nn


class SuperMLP(nn.Module):
    def __init__(self, emb_dim: int, hidden: Sequence[int],
                 activation: Callable,
                 activation_final: bool = False,
                 normalise: bool = False,
                 spectral_norm: bool = False,
                 residual: bool = False, name=None):
        super(SuperMLP, self).__init__()
        self._hidden = hidden
        self._activation = activation
        self._activation_final = activation_final
        self._normalise = normalise
        self._spectral_norm = spectral_norm
        self._residual = residual
        self.layers = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(emb_dim)
        for i in range(len(self._hidden) - 1):
            self.layers.append(nn.Linear(in_features=self._hidden[i], out_features=self._hidden[i + 1]))

    def forward(self, x, conditional=None, is_training=True):
        for i, size, layer in enumerate(zip(self._hidden, self.layers)):
            if conditional is not None:
                x = torch.cat([x, conditional], dim=-1)

            if i < len(self._hidden) - 1 or self._activation_final:
                if self._normalise:
                    h = self.layer_norm(h)
                h = self._activation(h)
            else:
                pass
            if self._residual:
                if size != x.shape[-1]:
                    x = layer(x)

                x += h
            else:
                x = h
        return x
