__all__ = ['euclidean_distance', 'cosine_similarity', 'get_num_samples', 'get_prototypes', 'prototypical_loss',
           'Encoder', 'Decoder', 'CAE', 'Encoder4L', 'Decoder4L', 'Decoder4L4Mini', 'CAE4L']

# adapted from the torchmeta code
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, return_indices=False, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2) if return_indices == False else nn.MaxPool2d(2, return_indices=True)
    )


def euclidean_distance2(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)

    return dist


def euclidean_distance(x, y):
    """
    x, y have shapes (batch_size, num_examples, embedding_size).
    x is prototypes, y are embeddings in most cases
    """
    return torch.sum((x.unsqueeze(2) - y.unsqueeze(1)) ** 2, dim=-1)


def sns(p, q):
    """
    Normally meant for query and prototypes
    :param q:
    :param p:
    :return:
    """
    p = F.normalize(p, dim=-1)
    sim = torch.einsum('bik, bjk->bij', p, q)
    return sim


def cosine_similarity(x, y):
    """x, y have shapes (batch_size, num_examples, embedding_size)."""

    # compute dot prod similarity x_i.T y_i (numerator)
    dot_similarity = torch.bmm(x, y.permute(0, 2, 1))

    # compute l2 norms ||x_i|| * ||y_i||
    x_norm = x.norm(p=2, dim=-1, keepdim=True)
    y_norm = y.norm(p=2, dim=-1, keepdim=True)

    norms = torch.bmm(x_norm, y_norm.permute(0, 2, 1)) + 1e-8

    return dot_similarity / norms


def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


def get_prototypes(emb, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each classes in the task.
    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor
        has shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has
        shape `(batch_size, num_examples)`.
    num_classes : int
        Number of classes in the task.
    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """

    batch_size, emb_size = emb.size(0), emb.size(-1)

    num_samples = get_num_samples(targets, num_classes, dtype=emb.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = emb.new_zeros((batch_size, num_classes, emb_size))
    indices = targets.unsqueeze(-1).expand_as(emb)

    prototypes.scatter_add_(1, indices, emb).div_(num_samples)

    return prototypes


def prototypical_loss(prototypes, embeddings, targets,
                      distance='euclidean', loss_fn=F.cross_entropy, temperature=1., **kwargs) -> Tuple[
    torch.Tensor, float, torch.Tensor]:
    """Compute the loss (i.e. negative log-likelihood) for the prototypical
    network, on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.

    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(batch_size, num_examples)`.

    distance : `String`
        The distance measure to be used: 'eucliden' or 'cosine'
    loss_fn : `Function`

    temperature : float

    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    if distance == 'euclidean':
        squared_distances = euclidean_distance(prototypes, embeddings)
        loss = loss_fn(-squared_distances / temperature, targets, **kwargs)
        _, predictions = torch.min(squared_distances, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
        logits = -squared_distances
    elif distance == 'cosine':
        cosine_similarities = cosine_similarity(prototypes, embeddings)
        loss = loss_fn(cosine_similarities, targets, **kwargs)
        _, predictions = torch.max(cosine_similarities, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
        logits = cosine_similarities
    elif distance == "sns":
        sns_similarities = sns(prototypes, embeddings)
        loss = loss_fn(sns_similarities, targets, **kwargs)
        _, predictions = torch.max(sns_similarities, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
        logits = sns_similarities
    else:
        raise ValueError('Distance must be "euclidean" or "cosine"')
    return loss, accuracy.item(), logits


class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 28x28 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class CAE(nn.Module):
    def __init__(self, in_channels, out_channels=64, hidden_size=64):
        super(CAE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = Encoder(num_input_channels=1, base_channel_size=64, latent_dim=64)

        self.decoder = Decoder(num_input_channels=1, base_channel_size=64, latent_dim=64)

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[-3:]))

        x = self.decoder(embeddings)
        return embeddings.view(*inputs.shape[:-3], -1), x.view(*inputs.shape)


class Encoder4L(nn.Module):
    def __init__(self, in_channels=1, hidden_size=64, out_channels=64):
        super().__init__()
        self.num_channels = out_channels

        self.encoder = nn.Sequential(
            # nn.ZeroPad2d(conv_padding),
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 x 14

            # nn.ZeroPad2d(conv_padding),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7

            # nn.ZeroPad2d(conv_padding),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 3x3

            # nn.ZeroPad2d(conv_padding),
            nn.Conv2d(hidden_size, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 1x1
            # nn.Flatten()
        )

    def forward(self, inputs):
        return self.encoder(inputs)


class Decoder4L(nn.Module):
    def __init__(self, in_channels=1, hidden_size=64, out_channels=64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(size=(4, 4)),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=hidden_size, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),

            nn.UpsamplingNearest2d(size=(7, 7)),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),

            nn.UpsamplingNearest2d(size=(14, 14)),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),

            nn.UpsamplingNearest2d(size=(28, 28)),
            nn.Conv2d(in_channels=hidden_size, out_channels=in_channels,
                      kernel_size=3, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.decoder(inputs)


class Decoder4L4Mini(nn.Module):
    def __init__(self, in_channels=3, hidden_size=64, out_channels=1600, mode='nearest'):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(size=(10, 10), mode=mode),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=hidden_size, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.GELU(),

            nn.Upsample(size=(21, 21), mode=mode),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.GELU(),

            nn.Upsample(size=(42, 42), mode=mode),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.GELU(),

            nn.Upsample(size=(84, 84), mode=mode),
            nn.Conv2d(in_channels=hidden_size, out_channels=in_channels,
                      kernel_size=3, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class CAE4L(nn.Module):
    def __init__(self, in_channels=1, hidden_size=64, out_channels=64):
        super().__init__()

        self.encoder = Encoder4L(in_channels=in_channels, hidden_size=hidden_size, out_channels=out_channels)
        self.decoder = Decoder4L(in_channels=in_channels, hidden_size=hidden_size, out_channels=out_channels)

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[-3:]))
        recons = self.decoder(embeddings.unsqueeze(-1).unsqueeze(-1))
        return embeddings.view(*inputs.shape[:-3], -1), recons.view(*inputs.shape)
