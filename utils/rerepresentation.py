import torch
import torch.nn.functional as F
from typing import Tuple

__all__ = ['re_represent']


def euclidean_distance(x, y):
    """
    x, y have shapes (batch_size, num_examples, embedding_size).
    x is prototypes, y are embeddings in most cases
    """
    return torch.sum((x.unsqueeze(2) - y.unsqueeze(1)) ** 2, dim=-1)


def re_represent(z: torch.Tensor, n_support: int,
                 alpha1: float, alpha2: float, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # being implemented with training shapes in mind
    # TODO: check if the same code works for testing shapes or requires some squeezing
    z_support = z[: n_support, :]
    z_query = z[n_support:, :]
    D = euclidean_distance(z_query.unsqueeze(0), z_query.unsqueeze(0)).squeeze(0)
    # D = torch.cdist(z_query, z_query).pow(2)
    A = F.softmax(t * D, dim=-1)
    scaled_query = (A.unsqueeze(-1) * z_query).sum(1)  # weighted sum of all query features
    z_query = (1 - alpha1) * z_query + alpha1 * scaled_query

    # Use re-represented query set to propagate information to the support set
    z_query = z_query.squeeze(0)
    D = euclidean_distance(z_support.unsqueeze(0), z_query.unsqueeze(0)).squeeze(0)
    # D = torch.cdist(z_support, z_query).pow(2)
    A = F.softmax(t * D, dim=-1)
    scaled_query = (A.unsqueeze(-1) * z_query).sum(1)
    z_support = (1 - alpha2) * z_support + alpha2 * scaled_query

    return z_support, z_query
