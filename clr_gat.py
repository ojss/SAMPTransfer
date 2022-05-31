__all__ = ['CLRGAT', 'GNN']

import copy
import uuid
from typing import Optional, Iterable, Union, Tuple

import einops
import kornia as K
import numpy as np
import pl_bolts.optimizers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.ops.adam import FusedAdam
from lightly.models.modules import NNCLRProjectionHead
from omegaconf import OmegaConf
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_metric_learning import losses
from torch.autograd import Variable
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

from dataloaders import UnlabelledDataModule
from graph.gat_v2 import GAT
from graph.gnn_base import GNNReID
from graph.graph_generator import GraphGenerator
from graph.latentgnn import LatentGNNV1
from proto_utils import (get_prototypes,
                         prototypical_loss)
from utils.optimal_transport import OptimalTransport
from utils.sk_finetuning import sinkhorned_finetuning
from utils.sup_finetuning import Classifier
from utils.label_cleansing import label_finetuning


class GNN(nn.Module):
    def __init__(self, backbone: nn.Module, emb_dim: int, mpnn_dev: str, mpnn_opts: dict, gnn_type: str = "gat"):
        super(GNN, self).__init__()
        self.backbone = backbone
        self.emb_dim = emb_dim
        self.mpnn_opts = mpnn_opts
        self.gnn_type = gnn_type
        mpnn_dev = mpnn_dev
        if gnn_type == "gat_v2":
            self.gnn = GAT(in_channels=emb_dim, hidden_channels=emb_dim // 4, out_channels=emb_dim,
                           num_layers=mpnn_opts["gnn_params"]["gnn"]["num_layers"],
                           heads=mpnn_opts["gnn_params"]["gnn"]["num_heads"],
                           v2=True, )
        elif gnn_type == "gat":
            self.gnn = GNNReID(mpnn_dev, mpnn_opts["gnn_params"], emb_dim)
        elif gnn_type == "latentgnn":
            self.gnn = LatentGNNV1(in_channels=64, latent_dims=[16, 16], channel_stride=2,
                                   num_kernels=2, mode="asymmetric",
                                   graph_conv_flag=False)
        self.graph_generator = GraphGenerator(mpnn_dev, **mpnn_opts["graph_params"])

    def forward(self, x):
        if "gat" in self.gnn_type:
            z = self.backbone(x).flatten(1)
            z_cnn = z.clone()
            edge_attr, edge_index, z = self.graph_generator.get_graph(z)
        else:
            z = self.backbone(x)
            z_cnn = z.clone().flatten(1)
        if self.gnn_type == "gat_v2":
            z = self.gnn(z, edge_index.t().contiguous())
        elif self.gnn_type == "gat":
            _, (z,) = self.gnn(z, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
        elif self.gnn_type == "latentgnn":
            z = self.gnn(z)
            z = z.flatten(1)
        return z_cnn, z


class CLRGAT(pl.LightningModule):
    def __init__(self,
                 n_support,
                 n_query,
                 batch_size,
                 lr_decay_step,
                 lr_decay_rate,
                 feature_extractor: nn.Module,
                 mpnn_loss_fn: Optional[Union[Optional[nn.Module], Optional[str]]],
                 mpnn_opts: dict,
                 mpnn_dev: str,
                 img_orig_size: Iterable,
                 label_cleansing_opts: dict,
                 gnn_type: str = "gat",
                 optim: str = 'adam',
                 dataset='omniglot',
                 weight_decay=0.01,
                 lr=1e-3,
                 lr_sch='cos',
                 warmup_epochs=10,
                 warmup_start_lr=1e-3,
                 eta_min=1e-5,
                 distance='euclidean',
                 mode='trainval',
                 eval_ways=5,
                 sup_finetune="prototune",
                 prototune_use_augs=False,
                 sup_finetune_lr=1e-3,
                 sup_finetune_epochs=15,
                 finetune_use_projector: Optional[int] = None,
                 ft_freeze_backbone=True,
                 finetune_batch_norm=False):
        super().__init__()
        self.save_hyperparameters()
        backbone = feature_extractor

        self.dataset = dataset
        self.batch_size = batch_size
        self.n_support = n_support
        self.n_query = n_query
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

        # PCLR Supfinetune
        self.mode = mode
        self.eval_ways = eval_ways
        self.sup_finetune = sup_finetune
        self.prototune_use_augs = prototune_use_augs
        self.sup_finetune_lr = sup_finetune_lr
        self.sup_finetune_epochs = sup_finetune_epochs
        self.ft_freeze_backbone = ft_freeze_backbone
        self.finetune_batch_norm = finetune_batch_norm
        self.finetune_use_projector = finetune_use_projector
        self.img_orig_size = img_orig_size

        self.label_cleansing_opts = label_cleansing_opts

        self.mpnn_opts = mpnn_opts
        _, in_dim = backbone(torch.randn(self.batch_size, 3, *img_orig_size)).flatten(1).shape

        self.dim = in_dim
        if mpnn_opts["_use"]:
            self.model = GNN(backbone, in_dim, mpnn_dev, mpnn_opts, gnn_type=gnn_type)
            self.mpnn_temperature = mpnn_opts["temperature"]
            if isinstance(mpnn_loss_fn, nn.Module):
                self.gnn_loss = mpnn_loss_fn
            elif mpnn_loss_fn == "ce":
                self.gnn_loss = F.cross_entropy
        else:
            self.model = backbone

        self.automatic_optimization = True

    def mpnn_forward_pass(self, x_support, x_query, y_support, y_query, ways):
        loss_cnn = 0.
        z_orig, z = self.mpnn_forward(torch.cat([x_support, x_query]),
                                      torch.cat([y_support, y_query], 1).squeeze())
        if self.mpnn_opts["loss_cnn"]:
            loss_cnn, _ = self.calculate_protoclr_loss(z_orig, y_support, y_query, ways,
                                                       temperature=self.mpnn_temperature)
            loss_cnn *= self.mpnn_opts["scaling_ce"]
            self.log("loss_cnn", loss_cnn.item())
        loss, acc = self.calculate_protoclr_loss(z, y_support, y_query,
                                                 ways, loss_fn=self.gnn_loss,
                                                 temperature=self.mpnn_temperature)
        loss = loss + loss_cnn
        return loss, acc, z

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

    def mpnn_forward(self, x, y=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: torch.Tensor
        :param y: torch.Tensor
        :return: Tuple(z_cnn, z)
        """
        z_cnn, z = self.model(x)

        return z_cnn, z

    def forward(self, x):
        if self.mpnn_opts["_use"]:
            _, z = self.model(x)
        else:
            z = self.model(x).flatten(1)
        return z

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

        loss, acc, z = self.mpnn_forward_pass(x_support, x_query, y_support, y_query, ways)
        self.log_dict({'loss': loss.item(), 'train_accuracy': acc}, prog_bar=True, on_epoch=True)

        return {"loss": loss, "accuracy": acc,
                "embeddings": z.detach()}  # accuracy return as 0 by default if CLR loss not used

    @torch.enable_grad()
    def prototune(self, episode, device='cpu', proto_init=True,
                  freeze_backbone=False, finetune_batch_norm=False,
                  inner_lr=0.001, total_epoch=15, n_way=5, use_augs=False):
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
        self.eval()
        proto = None
        if self.mpnn_opts["adapt"] == "task":
            z_support = self.model.backbone(x_a_i).flatten(1)
            z_query = self.model.backbone(x_b_i).flatten(1)
            nmb_proto = n_way
            z_proto = z_support.view(nmb_proto, n_support, -1).mean(1)
            combined = torch.cat([z_proto, z_query])
            edge_attr, edge_index, combined = self.graph_generator.get_graph(combined, Y=None)
            _, (combined,) = self.model.gnn(combined, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
            proto, query = combined.split([nmb_proto, len(z_query)])  # split based on number of prototypes
            z_a_i = z_support
        elif self.mpnn_opts["adapt"] == "proto_only":
            # instance level feature sharing
            combined = torch.cat([x_a_i, x_b_i])
            combined = self.model.backbone(combined).flatten(1)
            z_support, z_query = combined.split([n_support * n_way, len(x_b_i)])
            z_proto = z_support.view(n_way, n_support, -1).mean(1)
            edge_attr, edge_index, z_proto = self.graph_generator.get_graph(z_proto, Y=None)
            _, (z_proto,) = self.model.gnn(z_proto, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
            proto = z_proto
            z_a_i = z_support
        elif self.mpnn_opts["adapt"] == "instance":
            # TODO: change instance to include both x_a_i and x_b_i
            combined = torch.cat([x_a_i, x_b_i])
            _, combined = self.mpnn_forward(combined)
            z_a_i, _ = combined.split([len(x_a_i), len(x_b_i)])
        elif self.mpnn_opts["adapt"] == "ot":
            transportation_module = OptimalTransport(regularization=0.05, learn_regularization=False, max_iter=1000,
                                                     stopping_criterion=1e-4, device=self.device)
            z_a_i = self.forward(x_a_i)
            z_query = self.forward(x_b_i)
            z_a_i, _ = transportation_module(z_a_i, z_query)
        else:
            z_a_i = self.model.backbone(x_a_i).flatten(1)
        self.train()

        # Define projector layer
        if self.finetune_use_projector is not None:
            projector = NNCLRProjectionHead(1600, 1600, self.finetune_use_projector)
            input_dim = self.finetune_use_projector
        else:
            input_dim = z_a_i.shape[1]
        # Define linear classifier
        classifier = Classifier(input_dim, n_way=n_way)
        classifier.to(device)
        classifier.train()
        ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().to(device)
        sup_con_loss = losses.SupConLoss()
        if use_augs:
            augs = nn.Sequential(K.augmentation.ColorJitter(brightness=.4, contrast=.4, saturation=.4, hue=.1, p=0.8),
                                 K.augmentation.RandomResizedCrop(size=self.img_orig_size, scale=(0.5, 1.)),
                                 K.augmentation.RandomHorizontalFlip(),
                                 K.augmentation.RandomGrayscale(p=.2),
                                 K.augmentation.RandomGaussianBlur(kernel_size=(3, 3),
                                                                   sigma=(0.1, 2.0)))
        # Initialise as distance classifer (distance to prototypes)
        if proto_init:
            classifier.init_params_from_prototypes(z_a_i, n_way, n_support, z_proto=proto)
        if self.finetune_use_projector:
            parameters = list(classifier.parameters()) + list(projector.parameters())
        else:
            parameters = list(classifier.parameters())
        classifier_opt = torch.optim.Adam(parameters, lr=inner_lr)
        if freeze_backbone is False:
            delta_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.gnn.parameters()), lr=self.lr)
        # Finetuning
        if freeze_backbone is False:
            self.model.gnn.train()
        else:
            self.eval()
        classifier.train()
        if not finetune_batch_norm:
            for module in self.modules():
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
                if use_augs:
                    z_batch = torch.cat([z_batch, augs(z_batch)])
                    y_batch = y_a_i[selected_id].repeat(2)
                #####################################
                if self.mpnn_opts["adapt"] in ["task", "proto_only", "ot"]:
                    _, output = self.mpnn_forward(z_batch, y_batch)
                # TODO: implement instance level feature sharing by introducing the entire query set in forward pass
                elif self.mpnn_opts["adapt"] == "instance":
                    # lets use the entire query set?
                    combined = torch.cat([z_batch, x_b_i])
                    _, combined = self.mpnn_forward(combined)
                    output, _ = combined.split([len(z_batch), len(x_b_i)])
                else:
                    output = self.model.backbone(z_batch).flatten(1)

                if self.finetune_use_projector:
                    output = projector(output)
                preds = classifier(output)
                loss = loss_fn(preds, y_batch)
                if freeze_backbone is False or self.finetune_use_projector:
                    loss += sup_con_loss(output, y_batch)

                #####################################
                loss.backward()

                classifier_opt.step()

                if freeze_backbone is False:
                    delta_opt.step()
        classifier.eval()
        self.eval()

        y_query = torch.tensor(np.repeat(range(n_way), n_query)).to(self.device)
        if self.mpnn_opts["adapt"] == "task":
            # proto level feature sharing
            z_support = self.model.backbone(x_a_i).flatten(1)
            z_proto = z_support.view(nmb_proto, n_support, -1).mean(1)
            z_query = self.model.backbone(x_b_i).flatten(1)
            combined = torch.cat([z_proto, z_query])
            edge_attr, edge_index, combined = self.model.graph_generator.get_graph(combined, Y=None)
            _, (combined,) = self.model.gnn(combined, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
            proto, query = combined.split([nmb_proto, len(z_query)])
            output = query
        # cannot do proto adapt here
        elif self.mpnn_opts["adapt"] == "instance":
            combined = torch.cat([x_a_i, x_b_i])
            _, combined = self.mpnn_forward(combined)
            _, output = combined.split([len(x_a_i), len(x_b_i)])
        else:
            _, output = self.mpnn_forward(x_b_i)
        scores = classifier(output)

        loss = F.cross_entropy(scores, y_query, reduction='mean')
        _, predictions = torch.max(scores, dim=1)
        # acc = torch.mean(predictions.eq(y_query).float())
        acc = accuracy(predictions, y_query)
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
        x = einops.rearrange(x, "1 b c h w -> b c h w")
        if not (self.mpnn_opts["_use"] and self.mpnn_opts["adapt"] == "instance"):
            z = self.backbone(x)
            z = einops.rearrange(z, "b c h w -> 1 b (c h w)")
        else:
            _, z = self.mpnn_forward(x)
            z = einops.rearrange(z, "b e -> 1 b e")
        z_support = z[:, :self.eval_ways * shots]
        z_query = z[:, self.eval_ways * shots:]

        # Calucalte prototypes
        z_proto = get_prototypes(z_support, y_support, self.eval_ways)
        # implementing GAT based adaptation:
        if self.mpnn_opts["_use"] and self.mpnn_opts["adapt"] == "task":
            z_proto, z_query = einops.rearrange(z_proto, "1 b e -> b e"), einops.rearrange(z_query, "1 b e -> b e")
            combined = torch.cat([z_proto, z_query])
            edge_attr, edge_index, combined = self.graph_generator.get_graph(combined, Y=None)
            _, (combined,) = self.gnn(combined, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
            z_proto, z_query = combined.split([self.eval_ways, len(z_query)])  # split based on number of prototypes
            z_proto, z_query = einops.rearrange(z_proto, "b e -> 1 b e"), einops.rearrange(z_query, "b e -> 1 b e")
        elif self.mpnn_opts["_use"] and self.mpnn_opts["adapt"] == "proto_only":
            # adapt only the prototypes? like FEAT
            z_proto = einops.rearrange(z_proto, "1 b e -> b e")
            edge_attr, edge_index, z_proto = self.graph_generator.get_graph(z_proto)
            _, (z_proto,) = self.gnn(z_proto, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
            z_proto = einops.rearrange(z_proto, "b e -> 1 b e")

        # Calculate loss and accuracies
        loss, acc = prototypical_loss(z_proto, z_query, y_query,
                                      distance=self.distance)
        return loss, acc

    # TODO: check if validation is to be done with teacher or student

    def _shared_eval_step(self, batch, batch_idx):
        loss = 0.
        acc = 0.

        original_encoder_state = copy.deepcopy(self.state_dict())

        if self.sup_finetune == "prototune":
            loss, acc = self.prototune(
                episode=batch,
                inner_lr=self.sup_finetune_lr,
                total_epoch=self.sup_finetune_epochs,
                freeze_backbone=self.ft_freeze_backbone,
                finetune_batch_norm=self.finetune_batch_norm,
                device=self.device,
                n_way=self.eval_ways,
                use_augs=self.prototune_use_augs)
        elif self.sup_finetune == "label_cleansing":
            y_query, y_query_pred = self.lab_cleaning(batch, batch_idx)
            y_query, y_query_pred = [torch.Tensor(t) for t in [y_query, y_query_pred]]
            acc = accuracy(y_query_pred.long(), y_query.long())
            loss = torch.tensor(0.)  # because idk?
        elif self.sup_finetune == "std_proto":
            with torch.no_grad():
                loss, acc = self.std_proto_form(batch, batch_idx)
        elif self.sup_finetune == "sinkhorn":
            loss, acc = sinkhorned_finetuning(self, episode=batch, device=self.device, proto_init=True,
                                              freeze_backbone=self.ft_freeze_backbone,
                                              finetune_batch_norm=self.finetune_batch_norm, n_way=self.eval_ways,
                                              inner_lr=self.sup_finetune_lr)
        self.load_state_dict(original_encoder_state)
        return loss, acc

    @torch.enable_grad()
    def lab_cleaning(self, batch, batch_idx):
        x_support = batch['train'][0][0]  # only take data & only first batch
        x_support = x_support.to(self.device)
        x_support_var = Variable(x_support)
        x_query = batch['test'][0][0]  # only take data & only first batch
        x_query = x_query.to(self.device)
        x_query_var = Variable(x_query)
        n_support = x_support.shape[0] // self.eval_ways
        n_query = x_query.shape[0] // self.eval_ways

        batch_size = self.eval_ways
        support_size = self.eval_ways * n_support
        y_supp = Variable(torch.from_numpy(np.repeat(range(self.eval_ways), n_support))).to(self.device)
        y_query = torch.tensor(np.repeat(range(self.eval_ways), n_query)).to(self.device)
        z = self.forward(torch.cat([x_support_var, x_query_var]))
        support_features, query_features = z.split([len(x_support_var), len(x_query_var)])
        self.label_cleansing_opts["n_ways"] = self.eval_ways
        y_query, y_query_pred = label_finetuning(self.label_cleansing_opts, support_features, y_supp, y_query,
                                                 query_features)
        return y_query, y_query_pred

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


def cli_main():
    UUID = uuid.uuid4()
    OmegaConf.register_new_resolver("uuid", lambda: str(UUID))
    cli = LightningCLI(CLRGAT, UnlabelledDataModule, run=False,
                       save_config_overwrite=True,
                       parser_kwargs={"parser_mode": "omegaconf"})
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(ckpt_path=cli.trainer.checkpoint_callback.best_model_path, datamodule=cli.datamodule)


def slurm_main(conf_path, UUID):
    OmegaConf.register_new_resolver("uuid", lambda: str(UUID))
    print(conf_path)
    cli = LightningCLI(CLRGAT, UnlabelledDataModule, run=False,
                       save_config_overwrite=True,
                       save_config_filename=str(UUID),
                       parser_kwargs={"parser_mode": "omegaconf", "default_config_files": [conf_path]})
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(ckpt_path=cli.trainer.checkpoint_callback.best_model_path, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
