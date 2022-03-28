__all__ = ['Classifier', 'PCLROBoW']

import copy
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.cli import LightningCLI
from torch.autograd import Variable
from tqdm.auto import tqdm

import bow.bow_utils as utils
from bow.bow_extractor import BoWExtractorMultipleLevels
from bow.bowpredictor import BoWPredictor
from bow.classification import PredictionHead
from dataloaders import UnlabelledDataModule
from proto_utils import (Encoder4L,
                         get_prototypes,
                         prototypical_loss)


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


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

    def _set_params(self, weight, bias):
        state_dict = dict(weight=weight, bias=bias)
        self.fc.load_state_dict(state_dict)

    def init_params_from_prototypes(self, z_support, n_way, n_support):
        z_support = z_support.contiguous()
        z_proto = z_support.view(n_way, n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        # Interpretation of ProtoNet as linear layer (see Snell et al. (2017))
        self._set_params(weight=2 * z_proto, bias=-torch.norm(z_proto, dim=-1) ** 2)


# Cell
class PCLROBoW(pl.LightningModule):
    def __init__(self,
                 n_support,
                 n_query,
                 batch_size,
                 lr_decay_step,
                 lr_decay_rate,
                 bow_levels,
                 bow_extractor_opts_list,
                 bow_predictor_opts,
                 alpha=.99,
                 num_classes=None,
                 dataset='omniglot',
                 num_channels=64,  # number of output channels
                 feature_extractor=Encoder4L(3, 64, 64),
                 lr=1e-3,
                 inner_lr=1e-3,
                 distance='euclidean',
                 mode='trainval',
                 eval_ways=5,
                 sup_finetune=True,
                 sup_finetune_lr=1e-3,
                 sup_finetune_epochs=15,
                 ft_freeze_backbone=True,
                 finetune_batch_norm=False):
        super().__init__()
        assert isinstance(bow_levels, (list, tuple))
        assert isinstance(bow_extractor_opts_list, (list, tuple))
        assert len(bow_extractor_opts_list) == len(bow_levels)

        self._bow_levels = bow_levels
        self._num_bow_levels = len(bow_levels)
        if isinstance(alpha, (tuple, list)):
            # Use cosine schedule in order to increase the alpha from
            # alpha_base (e.g., 0.99) to 1.0.
            alpha_base, num_iterations = alpha
            self._alpha_base = alpha_base
            self._num_iterations = num_iterations
            self.register_buffer("_alpha", torch.FloatTensor(1).fill_(alpha_base))
            self.register_buffer("_iteration", torch.zeros(1))
            self._alpha_cosine_schedule = True
        else:
            self._alpha = alpha
            self._alpha_cosine_schedule = False

        # Building the student network
        self.feature_extractor = feature_extractor
        assert "kappa" in bow_predictor_opts
        bow_predictor_opts["num_channels_out"] = num_channels
        bow_predictor_opts["num_channels_hidden"] = num_channels * 2
        bow_predictor_opts["num_channels_in"] = [
            d["num_channels"] for d in bow_extractor_opts_list]
        self.bow_predictor = BoWPredictor(**bow_predictor_opts)

        # Building teacher network
        self.feature_extractor_teacher = copy.deepcopy(self.feature_extractor)
        self.bow_extractor = BoWExtractorMultipleLevels(bow_extractor_opts_list)

        if num_classes is not None:
            self.linear_classifier = PredictionHead(
                num_channels=num_channels, num_classes=num_classes,
                batch_norm=True, pool_type="global_avg"
            )
        else:
            self.linear_classifier = None

        for param, param_teacher in zip(self.feature_extractor.parameters(),
                                        self.feature_extractor_teacher.parameters()):
            param_teacher.data.copy_(param.data)  # initialise with the same weights
            param_teacher.requires_grad = False  # doesn't get updated by grades instead uses EWMA

        self.dataset = dataset
        self.batch_size = batch_size
        self.n_support = n_support
        self.n_query = n_query

        self.distance = distance

        # gamma will be used to weight the values of the MSE loss to potentially bring it up to par
        # gamma can also be adaptive in the future

        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.inner_lr = inner_lr

        self.mode = mode
        self.eval_ways = eval_ways
        self.sup_finetune = sup_finetune
        self.sup_finetune_lr = sup_finetune_lr
        self.sup_finetune_epochs = sup_finetune_epochs
        self.ft_freeze_backbone = ft_freeze_backbone
        self.finetune_batch_norm = finetune_batch_norm

        # self.example_input_array = [batch_size, 1, 28, 28] if dataset == 'omniglot'\
        #     else [batch_size, 3, 84, 84]

        self.automatic_optimization = False

    @torch.no_grad()
    def _get_momentum_alpha(self):
        if self._alpha_cosine_schedule:
            scale = 0.5 * (1. + math.cos((math.pi * self._iteration.item()) / self._num_iterations))
            self._alpha.fill_(1. - (1. - self._alpha_base) * scale)
            self._iteration += 1
            return self._alpha.item()
        else:
            return self._alpha

    @torch.no_grad()
    def _update_teacher(self):
        """ Exponetial moving average for the feature_extractor_teacher params:
            param_teacher = param_teacher * alpha + param * (1-alpha)
        """
        if not self.training:
            return
        alpha = self._get_momentum_alpha()
        if alpha >= 1.:
            return
        for param, param_teacher in zip(self.feature_extractor.parameters(),
                                        self.feature_extractor_teacher.parameters()):
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
            loss_cls, accuracy = self._linear_classifier(features, labels)
        return loss_cls, accuracy

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=.01)
        sch = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.lr_decay_step, gamma=self.lr_decay_rate)
        return {'optimizer': opt, 'lr_scheduler': sch}

    def forward(self, x):
        z = self.encoder(x.view(-1, *x.shape[-3:]))
        embeddings = nn.Flatten()(z)
        return embeddings.view(*x.shape[:-3], -1)

    def calculate_protoclr_loss(self, z, y_support, y_query, ways):

        #
        # e.g. [1,50*n_support,*(3,84,84)]
        z_support = z[:, :ways * self.n_support]
        # e.g. [1,50*n_query,*(3,84,84)]
        z_query = z[:, ways * self.n_support:]
        # Get prototypes
        if self.n_support == 1:
            z_proto = z_support  # in 1-shot the prototypes are the support samples
        else:
            z_proto = get_prototypes(z_support, y_support, ways)

        loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        # [batch_size x ways x shots x image_dim]
        data = batch['data'].to(self.device)
        data = data.unsqueeze(0)
        # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
        batch_size = data.size(0)
        ways = data.size(1)

        # Divide into support and query shots
        x_support = data[:, :, :self.n_support]
        # e.g. [1,50*n_support,*(3,84,84)]
        x_support = x_support.reshape(
            (batch_size, ways * self.n_support, *x_support.shape[-3:]))
        x_query = data[:, :, self.n_support:]
        # e.g. [1,50*n_query,*(3,84,84)]
        x_query = x_query.reshape(
            (batch_size, ways * self.n_query, *x_query.shape[-3:]))

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
        x = torch.cat([x_support, x_query], 1)
        z = self.forward(x)

        opt.zero_grad()

        loss, accuracy = self.calculate_protoclr_loss(
            z, y_support, y_query, ways)

        self.log('clr_loss', loss.item(), prog_bar=True)
        # adding the pixelwise reconstruction loss at the end
        # it has been broadcasted such that each support source image is broadcasted thrice over the three
        # query set images - which are the augmentations of the support image

        self.manual_backward(loss)
        opt.step()
        sch.step()

        self.log_dict({
            'loss': loss.item(),
            'train_accuracy': accuracy
        }, prog_bar=True)

        return loss, accuracy

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

        x_b_i = x_query_var
        x_a_i = x_support_var
        encoder.eval()

        # TODO: remove view add in network
        z_a_i = nn.Flatten()(encoder(x_a_i.to(device)))  # .view(*x_a_i.shape[:-3], -1)
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
            classifier.init_params_from_prototypes(z_a_i, n_way, n_support)
        classifier_opt = torch.optim.Adam(classifier.parameters(), lr=inner_lr)
        if freeze_backbone is False:
            delta_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, encoder.parameters()), lr=inner_lr)
        # Finetuning
        if freeze_backbone is False:
            encoder.train()
        else:
            encoder.eval()
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

                output = nn.Flatten()(encoder(z_batch))
                output = classifier(output)
                loss = loss_fn(output, y_batch)

                #####################################
                loss.backward()

                classifier_opt.step()

                if freeze_backbone is False:
                    delta_opt.step()
        classifier.eval()
        encoder.eval()

        output = nn.Flatten()(encoder(x_b_i.to(device)))
        scores = classifier(output)

        y_query = torch.tensor(np.repeat(range(n_way), n_query)).to(device)
        loss = F.cross_entropy(scores, y_query, reduction='mean')
        _, predictions = torch.max(scores, dim=1)
        accuracy = torch.mean(predictions.eq(y_query).float())
        return loss, accuracy.item()

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
        z, _ = self.forward(x)
        z_support = z[:, :self.eval_ways * shots]
        z_query = z[:, self.eval_ways * shots:]

        # Calucalte prototypes
        z_proto = get_prototypes(z_support, y_support, self.eval_ways)

        # Calculate loss and accuracies
        loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)
        return loss, accuracy

    def validation_step(self, batch, batch_idx):
        loss = 0.
        accuracy = 0.
        original_encoder_state = copy.deepcopy(self.encoder.state_dict())

        if self.sup_finetune:
            loss, accuracy = self.supervised_finetuning(self.encoder,
                                                        episode=batch,
                                                        inner_lr=self.sup_finetune_lr,
                                                        total_epoch=self.sup_finetune_epochs,
                                                        freeze_backbone=self.ft_freeze_backbone,
                                                        finetune_batch_norm=self.finetune_batch_norm,
                                                        device=self.device,
                                                        n_way=self.eval_ways)
            self.encoder.load_state_dict(original_encoder_state)
        elif not self.sup_finetune:
            with torch.no_grad():
                loss, accuracy = self.std_proto_form(batch, batch_idx)

        self.log_dict({
            'val_loss': loss.detach(),
            'val_accuracy': accuracy
        }, prog_bar=True)

        return loss.item(), accuracy

    def test_step(self, batch, batch_idx):
        original_encoder_state = copy.deepcopy(self.encoder.state_dict())
        if self.sup_finetune:
            loss, accuracy = self.supervised_finetuning(
                self.encoder,
                episode=batch,
                inner_lr=self.sup_finetune_lr,
                total_epoch=self.sup_finetune_epochs,
                freeze_backbone=self.ft_freeze_backbone,
                finetune_batch_norm=self.finetune_batch_norm,
                n_way=self.eval_ways,
            )
            torch.cuda.empty_cache()
            self.encoder.load_state_dict(original_encoder_state)
        else:
            # Note: this is just using the standard protonet form
            with torch.no_grad():
                loss, accuracy = self.std_proto_form(batch, batch_idx)

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
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss.item(), accuracy


def cli_main():
    cli = LightningCLI(PCLROBoW, UnlabelledDataModule, run=False)
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    cli_main()
