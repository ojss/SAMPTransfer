__all__ = ['Classifier', 'PCLROBoW']

import copy
import math
import uuid
from typing import Optional, Iterable, Union

import numpy as np
import pl_bolts.optimizers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.ops.adam import FusedAdam
from omegaconf import OmegaConf
from torch.autograd import Variable
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

import bow.bow_utils as utils
import vicreg.utils as vic_utils
from bow.bow_extractor import BoWExtractorMultipleLevels
from bow.bowpredictor import BoWPredictor
from bow.classification import PredictionHead
from cli.custom_cli import MyCLI
from dataloaders import UnlabelledDataModule
from feature_extractors.feature_extractor import CNN_4Layer
from graph.gnn_base import GNNReID
from graph.graph_generator import GraphGenerator
from utils.proto_utils import (get_prototypes,
                               prototypical_loss)
from utils.sup_finetuning import Classifier


@torch.no_grad()
def compute_bow_perplexity(bow_target):
    """ Compute the per image and per batch perplexity of the bow_targets. """
    assert isinstance(bow_target, (list, tuple))

    perplexity_batch, perplexity_img = [], []
    for bow_target_level in bow_target:  # For each bow level.
        assert bow_target_level.dim() == 2
        # shape of bow_target_level: [batch_size x num_words]

        probs = F.normalize(bow_target_level, p=1, dim=1)
        perplexity_img_level = torch.exp(
            -torch.sum(probs * torch.log(probs + 1e-5), dim=1)).mean()

        bow_target_sum_all = bow_target_level.sum(dim=0)
        # Uncomment the following line if you want to compute the perplexity of
        # of the entire batch in case of distributed training.
        # bow_target_sum_all = utils.reduce_all(bow_target_sum_all)
        probs = F.normalize(bow_target_sum_all, p=1, dim=0)
        perplexity_batch_level = torch.exp(
            -torch.sum(probs * torch.log(probs + 1e-5), dim=0))

        perplexity_img.append(perplexity_img_level)
        perplexity_batch.append(perplexity_batch_level)

    perplexity_batch = torch.stack(perplexity_batch, dim=0).view(-1).tolist()
    perplexity_img = torch.stack(perplexity_img, dim=0).view(-1).tolist()

    return perplexity_batch, perplexity_img


def expand_target(target, prediction):
    """Expands the target in case of BoW predictions from multiple crops."""
    assert prediction.size(1) == target.size(1)
    batch_size_x_num_crops, num_words = prediction.size()
    batch_size = target.size(0)
    assert batch_size_x_num_crops % batch_size == 0
    num_crops = batch_size_x_num_crops // batch_size

    if num_crops > 1:
        target = target.unsqueeze(1).repeat(1, num_crops, 1).view(-1, num_words)

    return target


# Cell
class PCLROBoW(pl.LightningModule):
    def __init__(self,
                 n_support,
                 n_query,
                 batch_size,
                 lr_decay_step,
                 lr_decay_rate,
                 feature_extractor: nn.Module,
                 bow_levels: list,
                 bow_extractor_opts: dict,
                 bow_predictor_opts: dict,
                 clr_loss: bool,
                 bow_clr: bool,
                 clr_on_bow: bool,
                 mpnn_loss_fn: Optional[Union[Optional[nn.Module], Optional[str]]],
                 mpnn_opts: dict,
                 mpnn_dev: str,
                 vicreg_opts: dict,
                 img_orig_size: Iterable,
                 optim: str = 'adam',
                 alpha=.99,
                 num_classes=None,
                 dataset='omniglot',
                 weight_decay=0.01,
                 lr=1e-3,
                 lr_sch='cos',
                 warmup_epochs=10,
                 warmup_start_lr=1e-3,
                 eta_min=1e-5,
                 inner_lr=1e-3,
                 alpha_cosine=True,
                 distance='euclidean',
                 mode='trainval',
                 eval_ways=5,
                 sup_finetune=True,
                 sup_finetune_lr=1e-3,
                 sup_finetune_epochs=15,
                 ft_freeze_backbone=True,
                 finetune_batch_norm=False):
        super().__init__()
        self.save_hyperparameters()

        assert isinstance(bow_levels, (list, tuple))
        # if isinstance(feature_extractor, ResNet):
        num_channels = feature_extractor.num_channels
        self.feature_extractor = feature_extractor
        self.num_words = bow_extractor_opts["num_words"]

        bow_extractor_opts_list = self.bow_opts_converter(bow_extractor_opts, bow_levels, num_channels)[
            'bow_extractor_opts_list']

        assert isinstance(bow_extractor_opts_list, (list, tuple))
        assert len(bow_extractor_opts_list) == len(bow_levels)

        self._bow_levels = bow_levels
        self._num_bow_levels = len(bow_levels)
        if alpha_cosine is True:
            # Use cosine schedule in order to increase the alpha from
            # alpha_base (e.g., 0.99) to 1.0.
            alpha_base = alpha
            self._alpha_base = alpha_base
            self.register_buffer("_alpha", torch.FloatTensor(1).fill_(alpha_base))
            self.register_buffer("_iteration", torch.zeros(1))
            self._alpha_cosine_schedule = True
        else:
            self._alpha = alpha
            self._alpha_cosine_schedule = False

        # Building the student network
        self.feature_extractor = feature_extractor
        assert "kappa" in bow_predictor_opts
        if bow_clr:
            bow_predictor_opts["num_channels_out"] = num_channels
            bow_predictor_opts["num_channels_hidden"] = num_channels * 2
            bow_predictor_opts["num_channels_in"] = [
                d["num_channels"] for d in bow_extractor_opts_list]
            self.bow_predictor = BoWPredictor(**bow_predictor_opts)
        if bow_clr or clr_on_bow:
            self.bow_extractor = BoWExtractorMultipleLevels(bow_extractor_opts_list)

        # Building teacher network
        if not clr_on_bow and not mpnn_opts["_use"]:
            # TODO: graph only on one network for now
            # CLR on BoW to be implemented as just one network
            self.feature_extractor_teacher = copy.deepcopy(self.feature_extractor)
            for param, param_teacher in zip(self.feature_extractor.parameters(),
                                            self.feature_extractor_teacher.parameters()):
                param_teacher.data.copy_(param.data)  # initialise with the same weights
                param_teacher.requires_grad = False  # doesn't get updated by grades instead uses EWMA

        if num_classes is not None:
            self.linear_classifier = PredictionHead(
                num_channels=num_channels, num_classes=num_classes,
                batch_norm=True, pool_type="global_avg"
            )
        else:
            self.linear_classifier = None

        self.dataset = dataset
        self.batch_size = batch_size
        self.n_support = n_support
        self.n_query = n_query

        self.clr_loss = clr_loss
        self.bow_clr = bow_clr
        self.clr_on_bow = clr_on_bow
        self.distance = distance

        self.weight_decay = weight_decay
        self.optim = optim
        self.lr = lr
        self.lr_sch = lr_sch
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.inner_lr = inner_lr

        # PCLR Supfinetune
        self.mode = mode
        self.eval_ways = eval_ways
        self.sup_finetune = sup_finetune
        self.sup_finetune_lr = sup_finetune_lr
        self.sup_finetune_epochs = sup_finetune_epochs
        self.ft_freeze_backbone = ft_freeze_backbone
        self.finetune_batch_norm = finetune_batch_norm

        self.mpnn_opts = mpnn_opts
        if mpnn_opts["_use"]:
            _, in_dim = self.feature_extractor(torch.randn(self.batch_size, 3, *img_orig_size)).flatten(1).shape
            self.dim = in_dim
            self.gnn = GNNReID(mpnn_dev, mpnn_opts["gnn_params"], in_dim).to(self.device)
            self.graph_generator = GraphGenerator(mpnn_dev, **mpnn_opts["graph_params"])
            self.mpnn_temperature = mpnn_opts["temperature"]
            if isinstance(mpnn_loss_fn, nn.Module):
                self.gnn_loss = mpnn_loss_fn
            elif mpnn_loss_fn == "ce":
                self.gnn_loss = F.cross_entropy

        self.vicreg = vicreg_opts
        self.automatic_optimization = True

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            # try:
            # self._num_iterations = len(self.train_dataloader()) * self.trainer.max_epochs
            self._num_iterations = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs

    def mpnn_shared_step(self, x, y):
        z = self.feature_extractor(x)
        z_orig = z.clone().flatten(1)
        z = z.flatten(1)
        edge_attr, edge_index, z = self.graph_generator.get_graph(z, y)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        preds, z = self.gnn(z, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])

        return preds, z_orig, z

    def mpnn_forward_pass(self, x_support, x_query, y_support, y_query, ways):
        loss_cnn = 0.
        _, z_orig, z = self.mpnn_shared_step(torch.cat([x_support, x_query]),
                                             torch.cat([y_support, y_query], 1).squeeze())
        if self.mpnn_opts["loss_cnn"]:
            loss_cnn, _ = self.calculate_protoclr_loss(z_orig, y_support, y_query, ways,
                                                       temperature=self.mpnn_temperature)
            loss_cnn *= self.mpnn_opts["scaling_ce"]
            self.log("loss_cnn", loss_cnn.item())
        loss, acc = self.calculate_protoclr_loss(z[0], y_support, y_query,
                                                 ways, loss_fn=self.gnn_loss,
                                                 temperature=self.mpnn_temperature)
        loss = loss + loss_cnn
        return loss, acc, z[0]

    @torch.no_grad()
    def _get_momentum_alpha(self):
        if self._alpha_cosine_schedule:
            scale = 0.5 * (1. + math.cos((math.pi * self._iteration.item()) / self._num_iterations))
            self._alpha.fill_(1. - (1. - self._alpha_base) * scale)
            self._iteration += 1
            return self._alpha.item()
        else:
            return self._alpha

    def _vicreg_loss(self, x, y):
        # TODO: remove representation loss since BoW Loss already handles this
        # repr_loss = F.mse_loss(x, y)
        effective_batch_size = (self.n_support + self.n_query) * self.batch_size
        x = self.all_gather(x)
        y = self.all_gather(y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 1e-4)  # for numerical stability
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (effective_batch_size - 1)
        cov_y = (y.T @ y) / (effective_batch_size - 1)
        cov_loss = vic_utils.off_diagonal(cov_x).pow_(2).sum().div(
            # Since we are currently only operating on the BoW representations and not the encoder representations
            self.dim  # TODO: pull this out of BoW predictor opts
        ) + vic_utils.off_diagonal(cov_y).pow_(2).sum().div(self.dim)

        loss = self.vicreg['std_coeff'] * std_loss + self.vicreg['cov_coeff'] * cov_loss
        return loss

    @torch.no_grad()
    def _update_teacher(self):
        """ Exponetial moving average for the feature_extractor_teacher params:
            param_teacher = param_teacher * alpha + param * (1-alpha)

            In case of MoCo the alpha is not scaled using a cosine schedule
        """
        if not self.training:
            return
        alpha = self._get_momentum_alpha()
        self.log("alpha", alpha)
        if alpha >= 1.:
            return
        student_params = self.feature_extractor.parameters()
        teacher_params = self.feature_extractor_teacher.parameters()
        for param, param_teacher in zip(student_params, teacher_params):
            param_teacher.data.mul_(alpha).add_(param.detach().data, alpha=(1. - alpha))

    def _bow_loss(self, bow_prediction, bow_target):
        assert isinstance(bow_prediction, (list, tuple))
        assert isinstance(bow_target, (list, tuple))
        assert len(bow_prediction) == self._num_bow_levels
        assert len(bow_target) == self._num_bow_levels
        # Instead of using a custom made cross-entropy loss for soft targets,
        # we use the pytorch kl-divergence loss that is defined as the
        # cross-entropy plus the entropy of targets. Since there is no gradient
        # back-propagation from the targets, it is equivalent to cross entropy.

        loss = [F.kl_div(F.log_softmax(p, dim=1), expand_target(t, p), reduction='batchmean') for (p, t) in
                zip(bow_prediction, bow_target)]
        return torch.stack(loss).mean()

    def _linear_classifier(self, features, labels):
        # With .detach() no gradients of the classification loss are
        # back-propagated to the feature extractor.
        # The reason for training such a linear classifier is in order to be
        # able to monitor while training the quality of the learned features.
        features = features.detach()
        if (labels is None) or (self.linear_classifier is None):
            return (features.new_full((1,), 0.0).squeeze(),
                    features.new_full((1,), 0.0).squeeze())
        scores = self.linear_classifier(features)
        loss = F.cross_entropy(scores, labels)
        with torch.no_grad():
            accuracy = utils.top1accuracy(scores, labels).item()

        return loss, accuracy

    def generate_bow_targets(self, image):
        # TODO: see if this can be updated for Conv-4 -> what levels to go for?
        if self.clr_on_bow:
            features = self.feature_extractor(image, self._bow_levels)
        else:
            features = self.feature_extractor_teacher(image, self._bow_levels)
        if isinstance(features, torch.Tensor):
            features = [features, ]
        bow_target, _ = self.bow_extractor(features)
        return bow_target, features

    def forward_test(self, img_orig, labels):
        with torch.no_grad():
            features = self.feature_extractor_teacher(img_orig, self._bow_levels)
            features = features if isinstance(features, torch.Tensor) else features[-1]
            features = features.detach()
            loss_cls, acc = self._linear_classifier(features, labels)
        return loss_cls, acc

    def configure_optimizers(self):
        # TODO: make this bit configurable
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        ret = {}
        if self.optim == 'sgd':
            opt = torch.optim.SGD(parameters, lr=self.lr, momentum=.9, weight_decay=self.weight_decay, nesterov=False)
        elif self.optim == 'adam':
            if torch.cuda.is_available():
                opt = FusedAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            else:
                opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim == 'radam':
            opt = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        ret["optimizer"] = opt

        if self.lr_sch == 'cos':
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.trainer.max_epochs)
            ret = {'optimizer': opt, 'lr_scheduler': sch}
        elif self.lr_sch == 'cos_warmup':
            sch = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(opt,
                                                                    warmup_epochs=self.warmup_epochs,
                                                                    max_epochs=self.trainer.max_epochs,
                                                                    warmup_start_lr=self.warmup_start_lr,
                                                                    eta_min=self.eta_min)
            ret = {'optimizer': opt, 'lr_scheduler': sch}
        elif self.lr_sch == 'step':
            sch = torch.optim.lr_scheduler.StepLR(opt, step_size=self.lr_decay_step, gamma=self.lr_decay_rate)
            ret['lr_scheduler'] = {'scheduler': sch, 'interval': 'step'}
        elif self.lr_sch == "one_cycle":
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr,
                                                      steps_per_epoch=self.trainer.limit_train_batches,
                                                      epochs=self.trainer.max_epochs)
            ret['lr_scheduler'] = {'scheduler': sch, 'interval': 'step'}
        return ret

    def forward(self, x, x_prime, labels=None):
        """ Applies the OBoW self-supervised task to a mini-batch of images.

        Args:
        x: 4D tensor with shape [batch_size x 3 x img_height x img_width]
            with the mini-batch of images from which the teacher network
            generates the BoW targets.
        x_prime: list of 4D tensors where each of them is a mini-batch of
            image crops with shape [(batch_size * num_crops) x 3 x crop_height x crop_width]
            from which the student network predicts the BoW targets. For
            example, in the full version of OBoW this list will include a
            [(batch_size * 2) x 3 x 160 x 160]-shaped tensor with two image crops
            of size [160 x 160] pixels and a [(batch_size * 5) x 3 x 96 x 96]-
            shaped tensor with five image patches of size [96 x 96] pixels.
        labels: (optional) 1D tensor with shape [batch_size] with the class
            labels of the img_orig images. If available, it would be used for
            on-line monitoring the performance of the linear classifier.

        Returns:
        losses: a tensor with the losses for each type of image crop and
            (optionally) the loss of the linear classifier.
        logs: a list of metrics for monitoring the training progress. It
            includes the perplexity of the bow targets in a mini-batch
            (perp_b), the perplexity of the bow targets in an image (perp_i),
            and (optionally) the accuracy of a linear classifier on-line
            trained on the teacher features (this is a proxy for monitoring
            during training the quality of the learned features; Note, at the
            end the features that are used are those of the student).
        """
        # TODO: implement new forward func
        if self.training is False:
            return

        ###### MAKE BOW PREDICTIONS ######
        dictionary = self.bow_extractor.get_dictionary()
        features = [self.feature_extractor(crop) for crop in x_prime]
        bow_predictions = self.bow_predictor(features, dictionary)

        ###### COMPUTE BOW TARGETS ######
        with torch.no_grad():
            self._update_teacher()
            bow_target, features_t = self.generate_bow_targets(x)
            perp_b, perp_i = compute_bow_perplexity(bow_target)

        ######## COMPUTE BOW PREDICTION LOSSES #######
        losses = [self._bow_loss(pred, bow_target) for pred in bow_predictions]

        # TODO: add classification loss later on
        losses = torch.stack(losses, dim=0).view(-1)
        logs = list(perp_b + perp_i)

        # TODO: change this to only use teacher for the training when using bow_clr
        if self.bow_clr or self.vicreg["use_vicreg"]:
            bow_predictions_x = self.bow_predictor([self.feature_extractor(x)], dictionary)
            feats = torch.cat([bow_predictions_x[0][0], bow_predictions[0][0]])
        else:
            features_o_stu = self.feature_extractor(x)
            feats = torch.cat([features_o_stu.flatten(1), features[0].flatten(1)])

        return losses, logs, feats

    @torch.enable_grad()
    def _clr_on_bow_forward(self, x_support, y_support, x_query, y_query, ways):
        # x_query = [x_query]
        # Compute BoW representations for all images
        bow_repr, z = self.generate_bow_targets(torch.cat([x_support, x_query]))
        z_supp_bow = bow_repr[0][:ways * self.n_support, ...].unsqueeze(0)
        z_query_bow = bow_repr[0][ways * self.n_support:, ...].unsqueeze(0)
        if self.n_support == 1:
            z_proto = z_supp_bow
        else:
            z_proto = get_prototypes(z_supp_bow, y_support, ways)
        loss, acc = prototypical_loss(z_proto, z_query_bow, y_query, distance=self.distance)
        return loss, acc, bow_repr[0]

    def calculate_protoclr_loss(self, z, y_support, y_query, ways, loss_fn=F.cross_entropy, temperature=1.):

        #
        # e.g. [1,50*n_support,*(3,84,84)]
        z_support = z[:ways * self.n_support, :].unsqueeze(0)
        # e.g. [1,50*n_query,*(3,84,84)]
        z_query = z[ways * self.n_support:, :].unsqueeze(0)
        # Get prototypes
        if self.n_support == 1:
            z_proto = z_support  # in 1-shot the prototypes are the support samples
        else:
            z_proto = get_prototypes(z_support, y_support, ways)

        loss, acc = prototypical_loss(z_proto, z_query, y_query,
                                      distance=self.distance, loss_fn=loss_fn, temperature=temperature)
        return loss, acc

    def training_step(self, batch, batch_idx):
        # [batch_size x ways x shots x image_dim]
        # data = batch['data'].to(self.device)
        acc = 0.
        data = batch['origs']
        views = batch['views']
        data = data.unsqueeze(0)
        # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
        batch_size = data.size(0)
        ways = data.size(1)

        # Divide into support and query shots
        # x_support = data[:, :, :self.n_support]
        # e.g. [1,50*n_support,*(3,84,84)]
        x_support = data.reshape(
            (batch_size, ways * self.n_support, *data.shape[-3:])).squeeze(0)
        x_query = views.reshape(
            (ways * self.n_query, *views.shape[-3:])
        )
        # x_query = data[:, :, self.n_support:].squeeze(0)
        # e.g. [1,50*n_query,*(3,84,84)]
        # x_query = x_query.reshape(
        #     (batch_size, ways * self.n_query, *x_query.shape[-3:]))

        # Create dummy query labels
        y_query = torch.arange(ways).unsqueeze(
            0).unsqueeze(2)  # batch and shot dim
        y_query = y_query.repeat(batch_size, 1, self.n_query)
        y_query = y_query.view(batch_size, -1).to(self.device)

        y_support = torch.arange(ways).unsqueeze(
            0).unsqueeze(2)  # batch and shot dim
        y_support = y_support.repeat(batch_size, 1, self.n_support)
        y_support = y_support.view(batch_size, -1).to(self.device)

        # Extract features (first dim is batch dim)
        # e.g. [1,50*(n_support+n_query),*(3,84,84)]
        # x = torch.cat([x_support, x_query], 1)
        if self.clr_on_bow:
            loss, acc, z = self._clr_on_bow_forward(x_support, y_support, x_query, y_query, ways)
        elif self.mpnn_opts["_use"]:
            loss, acc, z = self.mpnn_forward_pass(x_support, x_query, y_support, y_query, ways)
        else:
            losses, logs, z = self.forward(x_support, [x_query])

        if not self.clr_on_bow and not self.mpnn_opts["_use"]:
            loss = losses.sum()
            self.log("bow_loss", loss.item(), prog_bar=True)

        # Contrastive Loss based on ProtoCLR
        if self.clr_loss:
            clrl, acc = self.calculate_protoclr_loss(z, y_support, y_query, ways)
            self.log('clr_loss', clrl.item(), prog_bar=True)
            self.log("accuracy", acc, prog_bar=True)
            loss += clrl

        # Using only the VC losses of the VICReg loss. TODO: maybe see if the repr loss can be used?
        if self.vicreg["use_vicreg"]:
            vcl = self._vicreg_loss(z[:ways * self.n_support, ...], z[ways * self.n_support:, ...])
            self.log("vc_loss", vcl.item(), prog_bar=True)
            loss += vcl

        self.log_dict({'loss': loss.item(), 'train_accuracy': acc}, prog_bar=True, on_epoch=True)

        return {"loss": loss, "accuracy": acc,
                "embeddings": z.detach()}  # accuracy return as 0 by default if CLR loss not used

    @torch.enable_grad()
    def supervised_finetuning(self, encoder, episode, device='cpu', proto_init=True,
                              freeze_backbone=False, finetune_batch_norm=False,
                              inner_lr=0.001, total_epoch=15, n_way=5):
        x_support = episode['train'][0][0]  # only take data & only first batch
        x_support = x_support.to(device)
        x_support_var = Variable(x_support)
        x_query = episode['test'][0][0]  # only take data & only first batch
        x_query = x_query.to(device)
        x_query_var = Variable(x_query)
        n_support = x_support.shape[0] // n_way
        n_query = x_query.shape[0] // n_way

        batch_size = n_way
        support_size = n_way * n_support

        y_a_i = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).to(
            self.device)  # (25,)
        y_b_i = torch.tensor(np.repeat(range(n_way), n_query)).to(self.device)

        x_b_i = x_query_var
        x_a_i = x_support_var

        if self.mpnn_opts["_use"]:
            self.eval()
            proto = None
            if self.mpnn_opts["task_adapt"]:
                z_support = self.feature_extractor(x_a_i).flatten(1)
                z_query = self.feature_extractor(x_b_i).flatten(1)
                nmb_proto = n_way
                z_proto = z_support.view(nmb_proto, n_support, -1).mean(1)
                combined = torch.cat([z_proto, z_query])
                edge_attr, edge_index, combined = self.graph_generator.get_graph(combined, Y=None)
                _, (combined,) = self.gnn(combined, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
                proto, query = combined.split([nmb_proto, len(z_query)])  # split based on number of prototypes
                z_a_i = z_support
            else:
                _, _, (z_a_i,) = self.mpnn_shared_step(x_a_i, y_a_i)
            self.train()
        else:
            encoder.eval()
            z_a_i = encoder(x_a_i).flatten(1)
            encoder.train()

        # Define linear classifier
        input_dim = z_a_i.shape[1]
        classifier = Classifier(input_dim, n_way=n_way)
        classifier.to(device)
        classifier.train()
        ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().to(device)
        # Initialise as distance classifer (distance to prototypes)
        if proto_init:
            classifier.init_params_from_prototypes(z_a_i, n_way, n_support, z_proto=proto)
        classifier_opt = torch.optim.Adam(classifier.parameters(), lr=inner_lr)
        if freeze_backbone is False:
            delta_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, encoder.parameters()), lr=inner_lr)
            # TODO: add provisions for EdgeConv layer here until then freeze_backbone=False
        # Finetuning
        if freeze_backbone is False:
            encoder.train()
        else:
            encoder.eval()
            self.eval()
        classifier.train()
        if not finetune_batch_norm:
            for module in encoder.modules():
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()

        for epoch in tqdm(range(total_epoch), total=total_epoch, leave=False):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()
                if freeze_backbone is False:
                    delta_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy(
                    rand_id[j: min(j + batch_size, support_size)]).to(device)

                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id]
                #####################################
                if self.mpnn_opts["_use"]:
                    # TODO: can task adaptation be used here?
                    _, _, (output,) = self.mpnn_shared_step(z_batch, y_batch)
                else:
                    output = encoder(z_batch).flatten(1)

                output = classifier(output)
                loss = loss_fn(output, y_batch)

                #####################################
                loss.backward()

                classifier_opt.step()

                if freeze_backbone is False:
                    delta_opt.step()
        classifier.eval()
        self.eval()

        y_query = torch.tensor(np.repeat(range(n_way), n_query)).to(self.device)
        if self.mpnn_opts["_use"]:
            if self.mpnn_opts["task_adapt"]:
                # proto level feature sharing
                z_support = self.feature_extractor(x_a_i)
                z_proto = z_support.view(nmb_proto, n_support, -1).mean(1)
                z_query = self.feature_extractor(x_b_i).flatten(1)
                z_support = z_support.flatten(1)
                combined = torch.cat([z_proto, z_query])
                edge_attr, edge_index, combined = self.graph_generator.get_graph(combined, Y=None)
                _, (combined,) = self.gnn(combined, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
                proto, query = combined.split([nmb_proto, len(z_query)])
                output = query
            else:
                # instance level feature sharing
                combined = torch.cat([x_a_i, x_b_i])
                _, _, (combined,) = self.mpnn_shared_step(combined, y_query)
                _, output = combined.split([n_support * n_way, len(x_b_i)])
        else:
            output = encoder(x_b_i).flatten(1)

        scores = classifier(output)

        loss = F.cross_entropy(scores, y_query, reduction='mean')
        _, predictions = torch.max(scores, dim=1)
        # acc = torch.mean(predictions.eq(y_query).float())
        acc = accuracy(scores, y_query)
        return loss, acc.item()

    def std_proto_form(self, batch, batch_idx):
        x_support = batch["train"][0]
        y_support = batch["train"][1]
        x_support = x_support
        y_support = y_support

        x_query = batch["test"][0]
        y_query = batch["test"][1]
        x_query = x_query
        y_query = y_query

        # Extract shots
        shots = int(x_support.size(1) / self.eval_ways)
        test_shots = int(x_query.size(1) / self.eval_ways)

        # Extract features (first dim is batch dim)
        x = torch.cat([x_support, x_query], 1)
        z = self.feature_extractor(x)
        z_support = z[:, :self.eval_ways * shots]
        z_query = z[:, self.eval_ways * shots:]

        # Calucalte prototypes
        z_proto = get_prototypes(z_support, y_support, self.eval_ways)

        # Calculate loss and accuracies
        loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)
        return loss, accuracy

    # TODO: check if validation is to be done with teacher or student

    def _shared_eval_step(self, batch, batch_idx):
        loss = 0.
        acc = 0.

        original_encoder_state = copy.deepcopy(self.state_dict())

        if self.sup_finetune:
            loss, acc = self.supervised_finetuning(self.feature_extractor,
                                                   episode=batch,
                                                   inner_lr=self.sup_finetune_lr,
                                                   total_epoch=self.sup_finetune_epochs,
                                                   freeze_backbone=self.ft_freeze_backbone,
                                                   finetune_batch_norm=self.finetune_batch_norm,
                                                   device=self.device,
                                                   n_way=self.eval_ways, )
            self.load_state_dict(original_encoder_state)

        elif not self.sup_finetune:
            with torch.no_grad():
                loss, acc = self.std_proto_form(batch, batch_idx)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log_dict({
            'val_loss': loss.detach(),
            'val_accuracy': acc
        }, prog_bar=True)

        return loss.item(), acc

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)

        self.log(
            "test_loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss.item(), acc

    def bow_opts_converter(self, bow_extractor_opts, bow_levels, num_channels):
        model_opts = {}
        bow_extractor_opts = bow_extractor_opts
        num_words = bow_extractor_opts["num_words"]
        inv_delta = bow_extractor_opts["inv_delta"]
        bow_levels = bow_levels
        num_bow_levels = len(bow_levels)
        if not isinstance(inv_delta, (list, tuple)):
            inv_delta = [inv_delta for _ in range(num_bow_levels)]
        if not isinstance(num_words, (list, tuple)):
            num_words = [num_words for _ in range(num_bow_levels)]

        bow_extractor_opts_list = []
        for i in range(num_bow_levels):
            bow_extr_this = copy.deepcopy(bow_extractor_opts)
            if isinstance(bow_extr_this["inv_delta"], (list, tuple)):
                bow_extr_this["inv_delta"] = bow_extr_this["inv_delta"][i]
            if isinstance(bow_extr_this["num_words"], (list, tuple)):
                bow_extr_this["num_words"] = bow_extr_this["num_words"][i]
            bow_extr_this["num_channels"] = num_channels if isinstance(self.feature_extractor,
                                                                       CNN_4Layer) else num_channels // (
                    2 ** (num_bow_levels - 1 - i))
            bow_extractor_opts_list.append(bow_extr_this)

        model_opts["bow_extractor_opts_list"] = bow_extractor_opts_list

        return model_opts


def cli_main():
    UUID = uuid.uuid4()
    OmegaConf.register_new_resolver("uuid", lambda: str(UUID))
    cli = MyCLI(PCLROBoW, UnlabelledDataModule, run=False,
                save_config_overwrite=True,
                parser_kwargs={"parser_mode": "omegaconf"})
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(ckpt_path=cli.trainer.checkpoint_callback.best_model_path, datamodule=cli.datamodule)


def slurm_main(conf_path, UUID):
    OmegaConf.register_new_resolver("uuid", lambda: str(UUID))
    print(conf_path)
    cli = MyCLI(PCLROBoW, UnlabelledDataModule, run=False,
                save_config_overwrite=True,
                save_config_filename=str(UUID),
                parser_kwargs={"parser_mode": "omegaconf", "default_config_files": [conf_path]})
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(ckpt_path=cli.trainer.checkpoint_callback.best_model_path, datamodule=cli.datamodule)
    # deleting tmp config file:
    # if os.path.isfile(conf_path):
    #     os.remove(conf_path)


if __name__ == "__main__":
    cli_main()
