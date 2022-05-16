import copy
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

from dataloaders import get_episode_loader
from feature_extractors.feature_extractor import create_model
from utils.sup_finetuning import supervised_finetuning as generic_sup_finetune

try:
    from jarviscloud import jarviscloud
except ImportError as e:
    pass


class SimSiam(pl.LightningModule):
    def __init__(self, arch: str, data_path: str, fsl_data_path: str,
                 batch_size: int, num_workers: int, lr: float, optimiser: str, scheduler: str,
                 input_size: int = 224):
        super(SimSiam, self).__init__()
        if arch == "conv4":
            self.backbone = create_model(
                dict(in_planes=3, out_planes=[96, 128, 256, 512], num_stages=4, average_end=True))
        elif arch in torchvision.models.__dict__.keys():
            net = torchvision.models.__dict__[arch](pretrained=False)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
        with torch.no_grad():
            in_dim = self.backbone(torch.randn(1, 3, 224, 224)).flatten(1).shape[-1]
        self.projection_head = SimSiamProjectionHead(in_dim, in_dim, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)
        self.criterion = NegativeCosineSimilarity()
        self.lr = lr

        self.data_path = data_path
        self.fsl_data_path = fsl_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.optimiser = optimiser
        self.scheduler = scheduler

        self.save_hyperparameters()

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
        scheduler = None
        if self.optimiser == "adam":
            if torch.cuda.is_available():
                optim = deepspeed.ops.adam.FusedAdam(self.parameters(), lr=self.lr)
            else:
                optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimiser == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.lr)
        if self.scheduler == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.trainer.max_epochs)
            return [optim], [scheduler]
        return optim

    def train_dataloader(self):

        miniimg = torchvision.datasets.ImageFolder(self.data_path)
        dataset = LightlyDataset.from_torch_dataset(miniimg)

        collate_fn = SimCLRCollateFunction(input_size=self.input_size)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True,
                                drop_last=True,
                                num_workers=self.num_workers)
        return dataloader

    def _shared_eval_step(self, batch, batch_idx):
        loss = 0.
        acc = 0.

        original_enc_state = copy.deepcopy(self.backbone.state_dict())

        loss, acc = generic_sup_finetune(self.backbone, episode=batch, device=self.device, inner_lr=1e-3,
                                         total_epoch=15, freeze_backbone=True, finetune_batch_norm=False, n_way=5)
        self.backbone.load_state_dict(original_enc_state)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log_dict({"val_loss": loss.item(), "val_acc": acc}, on_step=True, on_epoch=True)
        return loss.item(), acc

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log_dict({"test_loss": loss.item(), "test_acc": acc}, on_step=True, on_epoch=True)
        return loss.item(), acc

    def test_dataloader(self):
        dataloader_test = get_episode_loader("miniimagenet", self.fsl_data_path,
                                             ways=5,
                                             shots=5,
                                             test_shots=15,
                                             batch_size=1,
                                             split='test',
                                             shuffle=False,
                                             num_workers=2)
        return dataloader_test

    def val_dataloader(self):
        dataloader_val = get_episode_loader("miniimagenet", self.fsl_data_path,
                                            ways=5,
                                            shots=5,
                                            test_shots=15,
                                            batch_size=1,
                                            split='val',
                                            shuffle=False,
                                            num_workers=2)
        return dataloader_val


def cli_main():
    cli = LightningCLI(SimSiam, run=False, parser_kwargs=dict(parser_mode="omegaconf"), save_config_overwrite=True)
    cli.trainer.fit(cli.model)
    wandb.finish()
    if "jarviscloud" in sys.modules:
        time.sleep(90)  # sleep for 3 mins so wandb can finish up
        jarviscloud.pause()


if __name__ == "__main__":
    cli_main()
