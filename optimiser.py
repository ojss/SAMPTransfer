import torch
from torch import nn
from typing import Optional, Callable
from torch.optim.optimizer import Optimizer
import functorch as ft


def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1.)


class MomentumOptimizer(Optimizer):
    def __init__(self, params, loss_fn, learning_rate: float = .125,
                 momentum: float = .9, name: Optional[str] = None):
        defaults = dict(
            lr=learning_rate, momentum=momentum
        )
        self.f = loss_fn
        super(MomentumOptimizer, self).__init__(params, defaults)

    def combined_objective(self, y, x):
        self.f()

    def __setstate__(self, state):
        super(MomentumOptimizer, self).__setstate__(state)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p_fp32 = p.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['grad_norm'] = torch.zeros_like()
