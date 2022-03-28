# Code pulled straight from original OBoW repo
import torch
from torch import nn

from . import bow_utils as utils


class BoWPredictor(nn.Module):
    def __init__(
            self,
            num_channels_out=2048,
            num_channels_in=[1024, 2048],
            num_channels_hidden=4096,
            kappa=8,
            learn_kappa=False
    ):
        """ Builds the dynamic BoW prediction head of the student network.

        It essentially builds a weight generation module for each BoW level for
        which the student network needs to predict BoW. For example, in its
        full version, OBoW uses two BoW levels, one for conv4 of ResNet (i.e.,
        penultimate feature scale of ResNet) and one for conv5 of ResNet (i.e.,
        final feature scale of ResNet). Therefore, in this case, the dynamic
        BoW prediction head has two weight generation modules.

        Args:
        num_channels_in: a list with the number of input feature channels for
            each weight generation module. For example, if OBoW uses two BoW
            levels and a ResNet50 backbone, then num_channels_in should be
            [1024, 2048], where the first number is the number of channels of
            the conv4 level of ResNet50 and the second number is the number of
            channels of the conv5 level of ResNet50.
        num_channels_out: the number of output feature channels for the weight
            generation modules.
        num_channels_hidden: the number of feature channels at the hidden
            layers of the weight generator modules.
        kappa: scalar with scale coefficient for the output weight vectors that
            the weight generation modules produce.
        learn_kappa (default False): if True kappa is a learnable parameter.
        """
        super(BoWPredictor, self).__init__()

        assert isinstance(num_channels_in, (list, tuple))
        num_bow_levels = len(num_channels_in)

        generators = []
        for i in range(num_bow_levels):
            generators.append(nn.Sequential())
            generators[i].add_module(f"b{i}_l2norm_in", utils.L2Normalize(dim=1))
            generators[i].add_module(f"b{i}_fc", nn.Linear(num_channels_in[i], num_channels_hidden, bias=False))
            generators[i].add_module(f"b{i}_bn", nn.BatchNorm1d(num_channels_hidden))
            generators[i].add_module(f"b{i}_rl", nn.ReLU(inplace=True))
            generators[i].add_module(f"b{i}_last_layer", nn.Linear(num_channels_hidden, num_channels_out))
            generators[i].add_module(f"b{i}_l2norm_out", utils.L2Normalize(dim=1))
        self.layers_w = nn.ModuleList(generators)

        self.scale = nn.Parameter(
            torch.FloatTensor(num_bow_levels).fill_(kappa),
            requires_grad=learn_kappa)

    def forward(self, features, dictionary):
        """Dynamically predicts the BoW from the features of cropped images.

        During the forward pass, it gets as input a list with the features from
        each type of extracted image crop and a list with the visual word
        dictionaries of each BoW level. First, it uses the weight generation
        modules for producing from each dictionary level the weight vectors
        that would be used for the BoW prediction. Then, it applies the
        produced weight vectors of each dictionary level to the given features
        to compute the BoW prediction logits.

        Args:
        features: list of 2D tensors where each of them is a mini-batch of
            features (extracted from the image crops) with shape
            [(batch_size * num_crops) x num_channels_out] from which the BoW
            prediction head predicts the BoW targets. For example, in the full
            version of OBoW, in which it reconstructs BoW from (a) 2 image crops
            of size [160 x 160] and (b) 5 image patches of size [96 x 96], the
            features argument includes a 2D tensor of shape
            [(batch_size * 2) x num_channels_out] (extracted from the 2
            160x160-sized crops) and a 2D tensor of shape
            [(batch_size * 5) x num_channels_out] (extractted from the 5
            96x96-sized crops).
        dictionary: list of 2D tensors with the visual word embeddings
            (i.e., dictionaries) for each BoW level. So, the i-th item of
            dictionary has shape [num_words x num_channels_in[i]], where
            num_channels_in[i] is the number of channels of the visual word
            embeddings at the i-th BoW level.

        Output:
        logits_list: list of lists of 2D tensors. Specifically, logits_list[i][j]
            contains the 2D tensor of size [(batch_size * num_crops) x num_words]
            with the BoW predictions from features[i] for the j-th BoW level
            (made using the dictionary[j]).
        """
        assert isinstance(dictionary, (list, tuple))
        assert len(dictionary) == len(self.layers_w)

        weight = [gen(dict).t() for gen, dict in zip(self.layers_w, dictionary)]
        kappa = torch.split(self.scale, 1, dim=0)
        logits_list = [
            [torch.mm(f.flatten(1) * k, w) for k, w in zip(kappa, weight)]
            for f in features]

        return logits_list

    def extra_repr(self):
        kappa = self.scale.data
        s = f"(kappa, learnable={kappa.requires_grad}): {kappa.tolist()}"
        return s
