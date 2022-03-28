import torch
from torch import nn
import torch.nn.functional as F

from . import bow_utils as utils


class BoWExtractor(nn.Module):
    def __init__(
            self,
            num_words,
            num_channels,
            update_type="local_average",
            inv_delta=15,
            bow_pool="max"):
        """Builds a BoW extraction module for the teacher network.

        It builds a BoW extraction module for the teacher network in which the
        visual words vocabulary is on-line updated during training via a
        queue-based vocabular/dictionary of randomly sampled local features.

        Args:
        num_words: the number of visual words in the vocabulary/dictionary.
        num_channels: the number of channels in the teacher feature maps and
            visual word embeddings (of the vocabulary).
        update_type: with what type of local features to update the queue-based
            visual words vocabulary. Three update types are implemenented:
            (a) "no_averaging": to update the queue it samples with uniform
            distribution one local feature vector per image from the given
            teacher feature maps.
            (b) "global_averaging": to update the queue it computes from each
            image a feature vector by globally average pooling the given
            teacher feature maps.
            (c) "local_averaging" (default option): to update the queue it
            computes from each image a feature vector by first locally averaging
            the given teacher feature map with a 3x3 kernel and then samples one
            of the resulting feature vectors with uniform distribution.
        inv_delta: the base value for the inverse temperature that is used for
            computing the soft assignment codes over the visual words, used for
            building the BoW targets. If inv_delta is None, then hard assignment
            is used instead.
        bow_pool: (default "max") how to reduce the assignment codes to BoW
            vectors. Two options are supported, "max" for max-pooling and "avg"
            for average-pooling.
        """
        super(BoWExtractor, self).__init__()

        if inv_delta is not None:
            assert isinstance(inv_delta, (float, int))
            assert inv_delta > 0.0
        assert bow_pool in ("max", "avg")
        assert update_type in ("local_average", "global_average", "no_averaging")

        self._num_channels = num_channels
        self._num_words = num_words
        self._update_type = update_type
        self._inv_delta = inv_delta
        self._bow_pool = bow_pool
        self._decay = 0.99

        embedding = torch.randn(num_words, num_channels).clamp(min=0)
        self.register_buffer("_embedding", embedding)
        self.register_buffer("_embedding_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_track_num_batches", torch.zeros(1))
        self.register_buffer("_min_distance_mean", torch.ones(1) * 0.5)

    @torch.no_grad()
    def _update_dictionary(self, features):
        """Given a teacher feature map it updates the queue-based vocabulary."""
        assert features.dim() == 4
        if self._update_type in ("local_average", "no_averaging"):
            if self._update_type == "local_average":
                features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=0)
            features = features.flatten(2)
            batch_size, _, num_locs = features.size()
            index = torch.randint(0, num_locs, (batch_size,), device=features.device)
            index += torch.arange(batch_size, device=features.device) * num_locs
            selected_features = features.permute(0, 2, 1).reshape(batch_size * num_locs, -1)
            selected_features = selected_features[index].contiguous()
        elif self._update_type == "global_average":
            selected_features = utils.global_pooling(features, type="avg").flatten(1)

        assert selected_features.dim() == 2
        # Gather the selected_features from all nodes in the distributed setting.
        selected_features = utils.concat_all_gather(selected_features)

        # To simplify the queue update implementation, it is assumed that the
        # number of words is a multiple of the batch-size.
        assert self._num_words % selected_features.shape[0] == 0
        batch_size = selected_features.shape[0]
        # Replace the oldest visual word embeddings with the selected ones
        # using the self._embedding_ptr pointer. Note that each training step
        # self._embedding_ptr points to the older visual words.
        ptr = int(self._embedding_ptr)
        self._embedding[ptr:(ptr + batch_size), :] = selected_features
        # move the pointer.
        self._embedding_ptr[0] = (ptr + batch_size) % self._num_words

    @torch.no_grad()
    def get_dictionary(self):
        """Returns the visual word embeddings of the dictionary/vocabulary."""
        return self._embedding.detach().clone()

    @torch.no_grad()
    def _broadast_initial_dictionary(self):
        # Make sure every node in the distributed setting starts with the
        # same dictionary. Maybe this is not necessary and copying the buffers
        # across the models on all gpus is handled by nn.DistributedDataParallel
        embedding = self._embedding.data.clone()
        torch.distributed.broadcast(embedding, src=0)
        self._embedding.data.copy_(embedding)

    def forward(self, features):
        """Given a teacher feature maps, it generates BoW targets."""
        features = features[:, :, 1:-1, 1:-1].contiguous()

        # Compute distances between features and visual words embeddings.
        embeddings_b = self._embedding.pow(2).sum(1)
        embeddings_w = -2 * self._embedding.unsqueeze(2).unsqueeze(3)
        # dist = ||features||^2 + |embeddings||^2 + conv(features, -2 * embedding)
        dist = (features.pow(2).sum(1, keepdim=True) +
                F.conv2d(features, weight=embeddings_w, bias=embeddings_b))
        # dist shape: [batch_size, num_words, height, width]
        min_dist, enc_indices = torch.min(dist, dim=1)
        mu_min_dist = min_dist.mean()
        mu_min_dist = utils.reduce_all(mu_min_dist) / utils.get_world_size()

        if self.training:
            # exponential moving average update of self._min_distance_mean.
            self._min_distance_mean.data.mul_(self._decay).add_(
                mu_min_dist, alpha=(1. - self._decay))
            self._update_dictionary(features)
            self._track_num_batches += 1

        if self._inv_delta is None:
            # Hard assignment codes.
            codes = dist.new_full(list(dist.shape), 0.0)
            codes.scatter_(1, enc_indices.unsqueeze(1), 1)
        else:
            # Soft assignment codes.
            inv_delta_adaptive = self._inv_delta / self._min_distance_mean
            codes = F.softmax(-inv_delta_adaptive * dist, dim=1)

        # Reduce assignment codes to bag-of-word vectors with global pooling.
        bow = utils.global_pooling(codes, type=self._bow_pool).flatten(1)
        bow = F.normalize(bow, p=1, dim=1)  # L1-normalization.
        return bow, codes

    def extra_repr(self):
        str_options = (
            f"num_words={self._num_words}, num_channels={self._num_channels}, "
            f"update_type={self._update_type}, inv_delta={self._inv_delta}, "
            f"pool={self._bow_pool}, "
            f"decay={self._decay}, "
            f"track_num_batches={self._track_num_batches.item()}")
        return str_options


class BoWExtractorMultipleLevels(nn.Module):
    def __init__(self, opts_list):
        """Builds a BoW extractor for each BoW level."""
        super(BoWExtractorMultipleLevels, self).__init__()
        assert isinstance(opts_list, (list, tuple))
        self.bow_extractor = nn.ModuleList([
            BoWExtractor(**opts) for opts in opts_list])

    @torch.no_grad()
    def get_dictionary(self):
        """Returns the dictionary of visual words from each BoW level."""
        return [b.get_dictionary() for b in self.bow_extractor]

    def forward(self, features):
        """Given a list of feature levels, it generates multi-level BoWs."""
        assert isinstance(features, (list, tuple))
        assert len(features) == len(self.bow_extractor)
        out = list(zip(*[b(f) for b, f in zip(self.bow_extractor, features)]))
        return out
