import sys
import time
from typing import Iterable, Union

import deepspeed.ops
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from lightly.data import LightlyDataset, SimCLRCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import NNCLRPredictionHead, NNCLRProjectionHead, NNMemoryBankModule
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from feature_extractors.feature_extractor import create_model

try:
    from jarviscloud import jarviscloud
except ImportError as e:
    pass


class NNCLRDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, bsize: int, num_workers: int, input_size: int):
        super(NNCLRDataModule, self).__init__()
        self.data_path = data_path
        self.num_classes = 80
        self.name = "miniimagenet"

        self.bsize = bsize
        self.num_workers = num_workers
        self.input_size = input_size

    def train_dataloader(self):
        train_ds = torchvision.datasets.ImageFolder(self.data_path + "/train")
        val_ds = torchvision.datasets.ImageFolder(self.data_path + "/val")
        trainval_ds = ConcatDataset([train_ds, val_ds])
        dataset = LightlyDataset.from_torch_dataset(trainval_ds)
        collate_fn = SimCLRCollateFunction(input_size=self.input_size)
        dataloader = DataLoader(dataset, batch_size=self.bsize, collate_fn=collate_fn, shuffle=True, drop_last=True,
                                num_workers=self.num_workers
                                )
        return dataloader

    def val_dataloader(self):
        val_ds = torchvision.datasets.ImageFolder(self.data_path + "/val")
        dataloader = DataLoader(val_ds, batch_size=self.bsize, shuffle=True, drop_last=True,
                                num_workers=self.num_workers
                                )
        return dataloader


class NNCLR(pl.LightningModule):
    def __init__(self, arch: str, data_path: str,
                 bsize: int, num_workers: int, lr: float, optimiser: str, scheduler: str,
                 conv_4_out_planes: Union[Iterable, int] = 64,
                 input_size: int = 84,
                 use_projector: bool = True,
                 projection_out_dim: int = 256,
                 proj_hidden_dim: int = 2048,
                 prediction_hidden_dim: int = 4096):
        super(NNCLR, self).__init__()
        self.out_planes = conv_4_out_planes
        if arch == "conv4":
            self.backbone = create_model(
                dict(in_planes=3, out_planes=self.out_planes, num_stages=4, average_end=True))
        elif arch in torchvision.models.__dict__.keys():
            net = torchvision.models.__dict__[arch](pretrained=False)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.z_dim = self.backbone(torch.randn(1, 3, input_size, input_size)).flatten(1).shape[-1]
        self.projection_out_dim = self.z_dim // 4 if projection_out_dim is None else projection_out_dim
        # TODO: try without using the projection and prediction heads
        if use_projector:
            self.projection_head = NNCLRProjectionHead(self.z_dim, proj_hidden_dim, self.projection_out_dim)
        else:
            self.projection_head = nn.Identity()
        self.prediction_head = NNCLRPredictionHead(self.projection_out_dim,
                                                   prediction_hidden_dim,
                                                   self.projection_out_dim)
        self.memory_bank = NNMemoryBankModule(size=8192)

        self.criterion = NTXentLoss()

        self.data_path = data_path
        self.bsize = bsize
        self.num_workers = num_workers
        self.lr = lr
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.input_size = input_size

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=True)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        if self.optimiser == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            if self.optimiser == "adam":
                if torch.cuda.is_available():
                    optim = deepspeed.ops.adam.FusedAdam(self.parameters(), lr=self.lr)
                else:
                    optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.trainer.max_epochs)
            return [optim], [scheduler]
        return [optim]

    def train_dataloader(self):
        miniimg = torchvision.datasets.ImageFolder(self.data_path + "/train")
        miniimg_val = torchvision.datasets.ImageFolder(self.data_path + "/val")
        miniimg = ConcatDataset([miniimg, miniimg_val])
        collate_fn = SimCLRCollateFunction(input_size=self.input_size)
        dataset = LightlyDataset.from_torch_dataset(miniimg)
        dataloader = DataLoader(dataset, batch_size=self.bsize, collate_fn=collate_fn, shuffle=True, drop_last=True,
                                num_workers=self.num_workers
                                )
        return dataloader


def cli_main():
    cli = LightningCLI(NNCLR,
                       NNCLRDataModule,
                       run=False,
                       parser_kwargs=dict(parser_mode="omegaconf"),
                       save_config_overwrite=True)
    cli.trainer.fit(cli.model, cli.datamodule)
    wandb.finish()
    if "jarviscloud" in sys.modules:
        time.sleep(90)  # sleep for 3 mins so wandb can finish up
        jarviscloud.pause()


if __name__ == "__main__":
    cli_main()
