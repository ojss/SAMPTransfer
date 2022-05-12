import copy
import sys
import time
from typing import Any, Tuple

import deepspeed
import pytorch_lightning as pl
import torch
import torchvision.datasets
import wandb
from lightly.data import DINOCollateFunction, LightlyDataset
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torch.utils.data import DataLoader

from feature_extractors.feature_extractor import CNN_4Layer
from graph.gnn_base import GNNReID
from graph.graph_generator import GraphGenerator

try:
    from jarviscloud import jarviscloud
except ImportError as e:
    pass


class GNN(nn.Module):
    def __init__(self, backbone: nn.Module, emb_dim: int, mpnn_opts: dict, img_orig_size: Tuple):
        super(GNN, self).__init__()
        self.backbone = backbone
        self.emb_dim = emb_dim
        self.mpnn_opts = mpnn_opts
        mpnn_dev = mpnn_opts["mpnn_dev"]
        self.gnn = GNNReID(mpnn_dev, mpnn_opts["gnn_params"], emb_dim)
        self.graph_generator = GraphGenerator(mpnn_dev, **mpnn_opts["graph_params"])

    def forward(self, x):
        z = self.backbone(x).flatten(1)
        edge_attr, edge_index, z = self.graph_generator.get_graph(z)
        _, (z,) = self.gnn(z, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
        return z


class DINO(pl.LightningModule):

    def __init__(self, arch: str, data_path: str, batch_size: int, num_workers: int,
                 mpnn_opts: dict,
                 img_orig_size: Tuple = (224, 224),
                 adaptive_avg_pool: bool = False):
        super(DINO, self).__init__()
        if arch == "conv4":
            if adaptive_avg_pool:
                backbone = CNN_4Layer(in_channels=3, global_pooling=True, final_maxpool=False, ada_maxpool=False)
            else:
                backbone = CNN_4Layer(in_channels=3, global_pooling=False, final_maxpool=True, ada_maxpool=True)
        elif arch in torchvision.models.__dict__.keys():
            net = torchvision.models.__dict__[arch](pretrained=False)
            backbone = nn.Sequential(*list(net.children())[:-1])
        with torch.no_grad():
            emb_dim = backbone(torch.rand(1, 3, *img_orig_size)).flatten(1).shape[-1]
        if mpnn_opts["_use"]:
            backbone = GNN(backbone, emb_dim, mpnn_opts, img_orig_size)

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(emb_dim, emb_dim, 64, 2048, freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(emb_dim, emb_dim, 64, 2048)

        deactivate_requires_grad(self.teacher_head)
        deactivate_requires_grad(self.teacher_backbone)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.save_hyperparameters()

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x) -> Any:
        y = self.teacher_backbone(x).flatten(1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        update_momentum(self.student_backbone, self.teacher_backbone, m=.99)
        update_momentum(self.student_head, self.teacher_head, m=.99)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        # views = [F.pad(view, (16, 16, 16, 16), "constant") for view in views[2:]]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]

        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("loss", loss.item(), prog_bar=True, on_epoch=True, on_step=True, batch_size=self.batch_size)
        return loss

    def on_after_backward(self) -> None:
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        if torch.cuda.is_available():
            optim = deepspeed.ops.adam.FusedAdam(self.parameters(), lr=0.001)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optim, step_size=15000, gamma=0.5),
            "interval": "step",
            "frequency": 1,
        }
        return [optim], [scheduler]

    def train_dataloader(self):

        miniimg = torchvision.datasets.ImageFolder(self.data_path)
        dataset = LightlyDataset.from_torch_dataset(miniimg)

        collate_fn = DINOCollateFunction(global_crop_size=224, local_crop_size=96)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True,
                                drop_last=True,
                                num_workers=self.num_workers)
        return dataloader


def cli_main():
    cli = LightningCLI(DINO, run=False, parser_kwargs={"parser_mode": "omegaconf"}, save_config_overwrite=True)
    cli.trainer.fit(cli.model)
    wandb.finish()
    if "jarviscloud" in sys.modules:
        time.sleep(90)  # sleep for 3 mins so wandb can finish up
        jarviscloud.pause()


if __name__ == "__main__":
    cli_main()
