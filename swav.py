import sys
import time
from typing import Tuple

import deepspeed.ops.adam
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import wandb
from lightly.data import LightlyDataset
from lightly.data import SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVPrototypes, SwaVProjectionHead
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torch.utils.data import DataLoader

from feature_extractors.feature_extractor import CNN_4Layer, create_model
from utils.gnn_wrapper import GNN

try:
    from jarviscloud import jarviscloud
except ImportError as e:
    pass


class SwaV(pl.LightningModule):
    def __init__(self, arch: str, lr: float, data_path: str, batch_size: int, num_workers: int,
                 mpnn_opts: dict,
                 img_orig_size: Tuple = (224, 224), ):
        super(SwaV, self).__init__()
        if arch == "conv4":
            backbone = create_model(dict(in_planes=3, out_planes=[96, 128, 256, 512], num_stages=4, average_end=True))
        elif arch in torchvision.models.__dict__.keys():
            net = torchvision.models.__dict__[arch](pretrained=False)
            backbone = nn.Sequential(*list(net.children())[:-1])
        with torch.no_grad():
            emb_dim = backbone(torch.rand(1, 3, *img_orig_size)).flatten(1).shape[-1]
        if mpnn_opts["_use"]:
            backbone = GNN(backbone, emb_dim, mpnn_opts, img_orig_size)
        self.backbone = backbone
        self.lr = lr

        self.datapath = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.projection_head = SwaVProjectionHead(emb_dim, emb_dim, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=512)
        self.criterion = SwaVLoss()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = F.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        self.prototypes.normalize()
        crops, _, _ = batch
        multi_crop_features = [self.forward(x.to(self.device)) for x in crops]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log("loss", loss.item(), on_step=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        if torch.cuda.is_available():
            optim = deepspeed.ops.adam.FusedAdam(self.parameters(), lr=self.lr)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.trainer.max_epochs)
        return [optim], [scheduler]

    def train_dataloader(self):
        miniimg = torchvision.datasets.ImageFolder(self.datapath)
        dataset = LightlyDataset.from_torch_dataset(miniimg)

        collate_fn = SwaVCollateFunction()

        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True,
                                drop_last=True,
                                num_workers=self.num_workers)
        return dataloader


def cli_main():
    cli = LightningCLI(SwaV, run=False, parser_kwargs={"parser_mode": "omegaconf"}, save_config_overwrite=True)
    cli.trainer.fit(cli.model)
    wandb.finish()
    if "jarviscloud" in sys.modules:
        time.sleep(90)  # sleep for 3 mins so wandb can finish up
        jarviscloud.pause()


if __name__ == "__main__":
    cli_main()
