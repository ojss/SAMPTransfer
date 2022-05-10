import sys
import time

import deepspeed
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead
from lightly.models.modules import SimSiamProjectionHead
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torch.utils.data import DataLoader

from feature_extractors.feature_extractor import CNN_4Layer

try:
    from jarviscloud import jarviscloud
except ImportError as e:
    pass


class SimSiam(pl.LightningModule):
    def __init__(self, arch: str, data_path: str, batch_size: int, num_workers: int, adaptive_avg_pool: bool = False,
                 input_size: int = 224):
        super(SimSiam, self).__init__()
        if arch == "conv4":
            if adaptive_avg_pool:
                self.backbone = CNN_4Layer(in_channels=3, global_pooling=True, final_maxpool=False, ada_maxpool=False)
            else:
                self.backbone = CNN_4Layer(in_channels=3, global_pooling=False, )
        elif arch in torchvision.models.__dict__.keys():
            net = torchvision.models.__dict__[arch](pretrained=False)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
        with torch.no_grad():
            in_dim = self.backbone(torch.randn(1, 3, 224, 224)).flatten(1).shape[-1]
        self.projection_head = SimSiamProjectionHead(in_dim, in_dim, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)
        self.criterion = NegativeCosineSimilarity

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("loss", loss.item(), on_step=True, on_epoch=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        if torch.cuda.is_available():
            optim = deepspeed.ops.adam.FusedAdam(self.parameters(), lr=0.06)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=0.06)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optim, step_size=15000, gamma=0.5),
            "interval": "step",
            "frequency": 1,
        }
        return [optim], [scheduler]

    def train_dataloader(self):

        miniimg = torchvision.datasets.ImageFolder(self.data_path)
        dataset = LightlyDataset.from_torch_dataset(miniimg)

        collate_fn = SimCLRCollateFunction(input_size=self.input_size)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True,
                                drop_last=True,
                                num_workers=self.num_workers)
        return dataloader


def cli_main():
    cli = LightningCLI(SimSiam, run=False, parser_kwargs=dict(parser_mode="omegaconf"), save_config_overwrite=True)
    cli.trainer.fit(cli.model)
    wandb.finish()
    if "jarviscloud" in sys.modules:
        time.sleep(90)  # sleep for 3 mins so wandb can finish up
    jarviscloud.pause()
