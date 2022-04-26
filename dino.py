import copy
import uuid
from typing import Callable, Union, Tuple, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from deepspeed.ops.adam import FusedAdam
from omegaconf import OmegaConf
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torch.autograd import Variable
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

import graph.graph_generator
from bow.feature_extractor import CNN_4Layer
from cli.custom_cli import DINOCli
from dino_dataloader import FewShotDatamodule
from graph.gnn_base import GNNReID
from losses import HLoss
from protoclr_obow import Classifier


class DINO(pl.LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 lr: float,
                 loss_fn: nn.Module,
                 dim: int,
                 center_momentum: float,
                 param_momentum: float,
                 sup_finetune: bool,
                 sup_finetune_lr: float,
                 sup_finetune_epochs: int,
                 ft_freeze_backbone: bool,
                 finetune_batch_norm: bool,
                 finetune_task_adapt: bool,
                 eval_ways: int = 5
                 ):
        super(DINO, self).__init__()
        self.teacher = encoder
        self.student = copy.deepcopy(self.teacher)

        self.lr = lr
        self.loss_fn = loss_fn
        self.c_mom = center_momentum
        self.p_mom = param_momentum

        self.sup_finetune = sup_finetune
        self.sup_finetune_lr = sup_finetune_lr
        self.sup_finetune_epochs = sup_finetune_epochs
        self.ft_freeze_backbone = ft_freeze_backbone
        self.finetune_batch_norm = finetune_batch_norm
        self.finetune_task_adapt = finetune_task_adapt
        self.eval_ways = eval_ways

        if isinstance(self.loss_fn, str):
            if self.loss_fn == "hloss":
                self.loss_fn = HLoss()

        self.register_buffer("center", torch.zeros((1, dim)).float())

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.save_hyperparameters()

    def configure_optimizers(self):
        return FusedAdam(self.student.parameters(), lr=self.lr)

    def loss_calculation(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        o, v = batch
        s1, s2 = self.student(o), self.student(v)
        t1, t2 = self.teacher(o), self.teacher(v)

        loss = self.loss_fn(t1, s2, self.center) + self.loss_fn(t2, s1, self.center)
        empirical_center = F.normalize(
            torch.cat([t1, t2]).mean(dim=0, keepdims=True),
            dim=-1
        )

        return loss, empirical_center

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], *args) -> dict:
        loss, empirical_center = self.loss_calculation(batch)
        self.log("train_loss", value=loss, on_step=True, on_epoch=True, prog_bar=True)

        self.center = F.normalize(
            self.c_mom * self.center + (1 - self.c_mom) * empirical_center,
            dim=-1
        )

        for s_p, t_p in zip(self.student.parameters(), self.teacher.parameters()):
            t_p.data = self.p_mom * t_p.data + (1 - self.p_mom) * s_p.data

        return loss

    @torch.enable_grad()
    def supervised_finetuning(self, encoder, episode, device='cpu', proto_init=True,
                              freeze_backbone=False, finetune_batch_norm=False,
                              inner_lr=0.001, total_epoch=15, n_way=5, ec=None, fusion=None):
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

        if self.finetune_task_adapt:
            z = encoder(torch.cat([x_a_i, x_b_i]))
            z_a_i = z[:support_size, :]
        else:
            z_a_i = encoder(x_a_i)
        self.train()

        # Define linear classifier
        input_dim = z_a_i.shape[1]
        classifier = Classifier(input_dim, n_way=n_way)
        classifier.to(device)
        classifier.train()
        ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        # Initialise as distance classifer (distance to prototypes)
        if proto_init:
            classifier.init_params_from_prototypes(z_a_i, n_way, n_support)
        classifier_opt = FusedAdam(classifier.parameters(), lr=inner_lr)
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
                output = encoder(z_batch)

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

        output = encoder(torch.cat([x_a_i, x_b_i]))[support_size:, :]

        scores = classifier(output)

        loss = F.cross_entropy(scores, y_query, reduction='mean')
        _, predictions = torch.max(scores, dim=1)
        # acc = torch.mean(predictions.eq(y_query).float())
        acc = accuracy(predictions, y_query)
        return loss, acc.item()

    def _shared_eval_step(self, batch, batch_idx):
        loss = 0.
        acc = 0.
        ec = self.ec if hasattr(self, "ec") else nn.Identity()
        fusion = self.fusion if hasattr(self, "fusion") else nn.Identity()

        original_encoder_state = copy.deepcopy(self.state_dict())

        if self.sup_finetune:
            loss, acc = self.supervised_finetuning(self.teacher,
                                                   episode=batch,
                                                   inner_lr=self.sup_finetune_lr,
                                                   total_epoch=self.sup_finetune_epochs,
                                                   freeze_backbone=self.ft_freeze_backbone,
                                                   finetune_batch_norm=self.finetune_batch_norm,
                                                   device=self.device,
                                                   n_way=self.eval_ways,
                                                   ec=ec,
                                                   fusion=fusion)
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


class Model(nn.Module):
    def __init__(self, encoder='conv4', gnn=True, gnn_opts: dict = None, dev="cpu", img_orig_size=(84, 84)):
        super(Model, self).__init__()
        self.gnn_opts = gnn_opts
        if encoder == "conv4":
            self.backbone = CNN_4Layer(in_channels=3, global_pooling=False)
        if gnn:
            _, in_dim = self.backbone(torch.randn(1, 3, *img_orig_size)).flatten(1).shape
            self.graph_generator = graph.graph_generator.GraphGenerator(dev=dev, **gnn_opts["graph_params"])
            self.gnn = GNNReID(dev, gnn_opts["gnn_params"], in_dim).to(dev)

    def forward(self, x):
        z = self.backbone(x)
        z = z.flatten(1)
        edge_attr, edge_index, z = self.graph_generator.get_graph(z)
        _, z = self.gnn(z, edge_index, edge_attr, self.gnn_opts["output_train_gnn"])
        return z[0]


def cli_main():
    UUID = uuid.uuid4()
    OmegaConf.register_new_resolver("uuid", lambda: str(UUID))
    cli = DINOCli(DINO, FewShotDatamodule, save_config_overwrite=True, run=False,
                  parser_kwargs={"parser_mode": "omegaconf"})
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(ckpt_path=cli.trainer.checkpoint_callback.last_model_path, datamodule=cli.datamodule)


def slurm_main(conf_path, UUID):
    OmegaConf.register_new_resolver("uuid", lambda: str(UUID))
    cli = DINOCli(DINO, FewShotDatamodule,
                  run=False,
                  save_config_overwrite=True,
                  save_config_filename=str(UUID),
                  parser_kwargs={"parser_mode": "omegaconf", "default_config_files": [conf_path]})
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(ckpt_path=cli.trainer.checkpoint_callback.last_model_path, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
