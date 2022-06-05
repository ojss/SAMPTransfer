__all__ = ['NNProtoCLR']

import copy
import uuid
from typing import Optional, Iterable, Union, Tuple, List

import numpy as np
import pl_bolts.optimizers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from deepspeed.ops.adam import FusedAdam
from lightly.data import LightlyDataset, SimCLRCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import NNCLRProjectionHead, NNCLRPredictionHead, NNMemoryBankModule
from omegaconf import OmegaConf
from pytorch_lightning.utilities.cli import LightningCLI
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

from clr_gat import GNN
from dataloaders import get_episode_loader
from feature_extractors.feature_extractor import create_model, resnet12_wide, resnet12
from misc.dino_utils import get_rank
from proto_utils import get_prototypes, prototypical_loss
from utils.optimal_transport import OptimalTransport
from utils.sk_finetuning import sinkhorned_finetuning
from utils.sup_finetuning import Classifier, std_proto_form


class NNProtoCLRDataModule(pl.LightningDataModule):
    def __init__(self, dataset: str, data_path: str, fsl_data_path: str, bsize: int, num_workers: int, input_size: int,
                 eval_ways: int, eval_shots: int, eval_query_shots: int):
        super(NNProtoCLRDataModule, self).__init__()
        self.data_path = data_path
        self.num_classes = 80
        self.name = dataset

        self.bsize = bsize
        self.num_workers = num_workers
        self.input_size = input_size

        self.fsl_data_path = fsl_data_path
        self.eval_ways = eval_ways
        self.eval_shots = eval_shots
        self.eval_query_shots = eval_query_shots

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
        return get_episode_loader(self.name, self.fsl_data_path,
                                  ways=self.eval_ways,
                                  shots=self.eval_shots,
                                  test_shots=self.eval_query_shots,
                                  batch_size=1,
                                  split='val',
                                  shuffle=False)

    def test_dataloader(self):
        return get_episode_loader(self.name, self.fsl_data_path,
                                  ways=self.eval_ways,
                                  shots=self.eval_shots,
                                  test_shots=self.eval_query_shots,
                                  batch_size=1,
                                  split='test',
                                  num_workers=4,
                                  shuffle=False)


class NNProtoCLR(pl.LightningModule):
    def __init__(self,
                 batch_size,
                 lr_decay_step,
                 lr_decay_rate,
                 arch: Optional[str],
                 conv_4_out_planes: Optional[Union[List, int]],
                 mpnn_loss_fn: Optional[Union[Optional[nn.Module], Optional[str]]],
                 adapt: str,
                 mpnn_opts: dict,
                 mpnn_dev: str,
                 img_orig_size: Iterable,
                 queue_size: int,
                 temperature: float = 0.1,
                 n_support: int = 1,
                 n_query: int = 3,
                 use_projector: bool = True,
                 use_prediction_head: bool = True,
                 projection_out_dim: int = 256,
                 projection_h_dim: int = 2048,
                 pred_h_dim: int = 4096,
                 optim: str = 'adam',
                 dataset='omniglot',
                 weight_decay=0.01,
                 lr=1e-3,
                 lr_sch='cos',
                 classifier_lr: float = 1e-3,
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
                 ft_freeze_backbone=True,
                 finetune_batch_norm=False,
                 alpha1=.3,
                 alpha2=.3,
                 re_rep_temp=0.07):
        super().__init__()
        self.save_hyperparameters()
        self.out_planes = conv_4_out_planes
        if arch == "conv4":
            backbone = create_model(
                dict(in_planes=3, out_planes=self.out_planes, num_stages=4,
                     average_end=False if conv_4_out_planes == 64 else True))
        elif arch == "resnet12":
            backbone = resnet12()
        elif arch == "resnet12_wide":
            backbone = resnet12_wide()
        elif arch in torchvision.models.__dict__.keys():
            net = torchvision.models.__dict__[arch](pretrained=False)
            backbone = nn.Sequential(*list(net.children())[:-1])

        self.dataset = dataset
        self.batch_size = batch_size
        self.distance = distance

        self.weight_decay = weight_decay
        self.use_projector = use_projector
        self.optim = optim
        self.lr = lr
        self.lr_sch = lr_sch
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.n_support = n_support
        self.n_query = n_query
        self.temperature = temperature

        # PCLR Supfinetune
        self.adapt = adapt
        self.mode = mode
        self.eval_ways = eval_ways
        self.sup_finetune = sup_finetune
        self.prototune_use_augs = prototune_use_augs
        self.sup_finetune_lr = sup_finetune_lr
        self.sup_finetune_epochs = sup_finetune_epochs
        self.ft_freeze_backbone = ft_freeze_backbone
        self.finetune_batch_norm = finetune_batch_norm
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.re_rep_temp = re_rep_temp

        self.img_orig_size = img_orig_size

        self.queue_size = queue_size

        self.mpnn_opts = mpnn_opts
        _, emb_dim = backbone(torch.randn(self.batch_size, 3, *img_orig_size)).flatten(1).shape
        if mpnn_opts["_use"]:
            self.emb_dim = emb_dim
            self.model = GNN(backbone, emb_dim, mpnn_dev, mpnn_opts)
            self.mpnn_temperature = mpnn_opts["temperature"]
            if isinstance(mpnn_loss_fn, nn.Module):
                self.gnn_loss = mpnn_loss_fn
            elif mpnn_loss_fn == "ce":
                self.gnn_loss = F.cross_entropy
        else:
            self.model = backbone

        if use_projector:
            self.projection_head = NNCLRProjectionHead(emb_dim, projection_h_dim, projection_out_dim)
        else:
            self.projection_head = nn.Identity()
            projection_out_dim = emb_dim

        # How about doing away with Projection and Prediction heads
        if use_prediction_head:
            self.prediction_head = NNCLRPredictionHead(projection_out_dim, pred_h_dim, projection_out_dim)
        else:
            self.prediction_head = nn.Identity()

        # TODO: add online classifier later for training acc
        # self.classifier = nn.Linear(in_features=emb_dim, out_features=80)
        # self.classifier_lr = classifier_lr

        # queue

        self.memory_bank = NNMemoryBankModule(size=self.queue_size)
        self.criterion = NTXentLoss(temperature=self.temperature)
        self.w = 0.9
        self.register_buffer("avg_output_std", torch.tensor(0.))

        self.automatic_optimization = True

    @torch.no_grad()
    def dequeue_and_enqueue(self, z: torch.Tensor, y: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner. Also stores
        the labels of the samples.

        Args:
            z (torch.Tensor): batch of projected features.
            y (torch.Tensor): labels of the samples in the batch.
        """
        # TODO: handle distributed mode

        batch_size = z.shape[0]

        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0

        self.queue[ptr: ptr + batch_size, :] = z
        self.queue_y[ptr: ptr + batch_size] = y  # type: ignore
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr  # type: ignore

    @torch.no_grad()
    def find_nn(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds the nearest neighbor of a sample.

        Args:
            z (torch.Tensor): a batch of projected features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                indices and projected features of the nearest neighbors.
        """

        idx = (z @ self.queue.T).max(dim=1)[1]
        nearest_neighbour = self.queue[idx]
        return idx, nearest_neighbour

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

    @staticmethod
    def re_represent(z: torch.Tensor, n_support: int,
                     alpha1: float, alpha2: float, t: float):
        # being implemented with training shapes in mind
        # TODO: check if the same code works for testing shapes or requires some squeezing
        z_support = z[: n_support, :]
        z_query = z[n_support:, :]
        D = torch.cdist(z_query, z_query).pow(2)
        A = F.softmax(t * D, dim=-1)
        scaled_query = (A.unsqueeze(-1) * z_query).sum(1)  # weighted sum of all query features
        z_query = (1 - alpha1) * z_query + alpha1 * scaled_query

        # Use re-represented query set to propagate information to the support set
        D = torch.cdist(z_support, z_query).pow(2)
        A = F.softmax(t * D, dim=-1)
        scaled_query = (A.unsqueeze(-1) * z_query).sum(1)
        z_support = (1 - alpha2) * z_support + alpha2 * scaled_query

        return z_support, z_query

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        ret = {}
        if self.optim == 'sgd':
            opt = torch.optim.SGD(parameters, lr=self.lr, momentum=.9, weight_decay=self.weight_decay,
                                  nesterov=False)
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

    def mpnn_forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: torch.Tensor
        :return: Tuple(z_cnn, z)
        """
        z_cnn, z = self.model(x)

        return z_cnn, z

    def forward(self, x):
        if self.mpnn_opts["_use"]:
            _, y = self.model(x)
        else:
            y = self.model(x).flatten(1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return y, z, p

    def nnclr_loss_func(self, nn: torch.Tensor, p: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """Computes NNCLR's loss given batch of nearest-neighbors nn from view 1 and
        predicted features p from view 2.

        Args:
            nn (torch.Tensor): NxD Tensor containing nearest neighbors' features from view 1.
            p (torch.Tensor): NxD Tensor containing predicted features from view 2
            temperature (float, optional): temperature of the softmax in the contrastive loss. Defaults
                to 0.1.

        Returns:
            torch.Tensor: NNCLR loss.
        """

        nn = F.normalize(nn, dim=-1)
        p = F.normalize(p, dim=-1)
        # to be consistent with simclr, we now gather p
        # this might result in suboptimal results given previous parameters.

        logits = nn @ p.T / temperature

        rank = get_rank()
        n = nn.size(0)
        labels = torch.arange(n * rank, n * (rank + 1), device=p.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        (x0, x1), targets, _ = batch
        _o0, z0, p0 = self.forward(x0)
        _o1, z1, p1 = self.forward(x1)

        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=True)

        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))

        z0_std = F.normalize(z0, dim=-1).std(dim=0).mean()
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z_std = (z0_std + z1_std) / 2

        self.log("z_std", z_std, on_step=True, on_epoch=True)

        # o = p0.detach()
        # o = F.normalize(o, dim=-1)
        # o_std = torch.std(o, 0)
        # o_std = o_std.mean()
        # self.avg_output_std = self.w * self.avg_output_std + (1 - self.w) * o_std.item()
        # collapse_level = max(0., 1 - math.sqrt(128) * self.avg_output_std)
        # self.log("collapse_level", collapse_level, on_epoch=True)
        self.log_dict(dict(train_loss=loss.item()),  # TODO: add nn_acc back later
                      on_epoch=True,
                      on_step=True,
                      prog_bar=True)
        return loss

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

        y_a_i = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).to(self.device)  # (25,)
        y_b_i = torch.tensor(np.repeat(range(n_way), n_query)).to(self.device)

        x_b_i = x_query_var
        x_a_i = x_support_var
        self.eval()
        if self.adapt == "instance":
            assert self.mpnn_opts["_use"] is True
            # TODO: change instance to include both x_a_i and x_b_i
            combined = torch.cat([x_a_i, x_b_i])
            _, combined = self.mpnn_forward(combined)
            z_a_i, _ = combined.split([len(x_a_i), len(x_b_i)])
        elif self.adapt == "ot":
            transportation_module = OptimalTransport(regularization=0.05, learn_regularization=False, max_iter=1000,
                                                     stopping_criterion=1e-4, device=self.device)
            z_a_i, _, _ = self.forward(x_a_i)
            z_query, _, _ = self.forward(x_b_i)
            z_a_i, _ = transportation_module(z_a_i, z_query)
        elif self.adapt == "re_rep":
            combined = torch.cat([x_a_i, x_b_i])
            z, _, _ = self.forward(combined)
            z_a_i, z_b_i = self.re_represent(z, support_size, self.alpha1, self.alpha2, self.re_rep_temp)
        else:
            z_a_i = self.model(x_a_i).flatten(1)
        self.train()

        input_dim = z_a_i.shape[1]
        # Define linear classifier
        classifier = Classifier(input_dim, n_way=n_way)
        classifier.to(device)
        classifier.train()
        ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().to(device)
        # Initialise as distance classifer (distance to prototypes)
        if proto_init:
            classifier.init_params_from_prototypes(z_a_i, n_way, n_support, )
        classifier_opt = torch.optim.Adam(classifier.parameters(), lr=inner_lr)
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
                #####################################
                if self.adapt == "instance":
                    assert self.mpnn_opts["_use"] is True
                    # lets use the entire query set?
                    combined = torch.cat([z_batch, x_b_i])
                    _, combined = self.mpnn_forward(combined)
                    output, _ = combined.split([len(z_batch), len(x_b_i)])
                elif self.adapt == "re_rep":
                    combined = torch.cat([z_batch, x_b_i])
                    combined, _, _ = self.forward(combined)
                    output, _ = self.re_represent(combined, len(z_batch), self.alpha1, self.alpha2,
                                                  self.re_rep_temp)
                else:
                    output = self.model(z_batch).flatten(1)

                preds = classifier(output)
                loss = loss_fn(preds, y_batch)

                #####################################
                loss.backward()

                classifier_opt.step()

                if freeze_backbone is False:
                    delta_opt.step()
        classifier.eval()
        self.eval()

        y_query = torch.tensor(np.repeat(range(n_way), n_query)).to(self.device)

        if self.adapt == "instance":
            assert self.mpnn_opts["_use"]
            combined = torch.cat([x_a_i, x_b_i])
            _, combined = self.mpnn_forward(combined)
            _, output = combined.split([len(x_a_i), len(x_b_i)])
        else:
            # TODO: add re_rep here
            output, _, _ = self.forward(x_b_i)
        output = output.flatten(1)
        scores = classifier(output)

        loss = F.cross_entropy(scores, y_query, reduction='mean')
        _, predictions = torch.max(scores, dim=1)
        # acc = torch.mean(predictions.eq(y_query).float())
        acc = accuracy(predictions, y_query)
        return loss, acc.item()

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
        elif self.sup_finetune == "std_proto":
            with torch.no_grad():
                loss, acc = std_proto_form(self, batch, batch_idx)
        elif self.sup_finetune == "sinkhorn":
            loss, acc = sinkhorned_finetuning(self, episode=batch, device=self.device, proto_init=True,
                                              freeze_backbone=self.ft_freeze_backbone,
                                              finetune_batch_norm=self.finetune_batch_norm, n_way=self.eval_ways,
                                              inner_lr=self.sup_finetune_lr)
        self.load_state_dict(original_encoder_state)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log_dict({
            'val_loss': loss.detach(),
            'val_accuracy': acc
        }, prog_bar=True, on_step=True, on_epoch=True)

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
    cli = LightningCLI(NNProtoCLR, NNProtoCLRDataModule, run=False,
                       save_config_overwrite=True,
                       parser_kwargs={"parser_mode": "omegaconf"})
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(ckpt_path=cli.trainer.checkpoint_callback.last_model_path, datamodule=cli.datamodule)


def slurm_main(conf_path, UUID):
    OmegaConf.register_new_resolver("uuid", lambda: str(UUID))
    print(conf_path)
    cli = LightningCLI(NNProtoCLR, NNProtoCLRDataModule, run=False,
                       save_config_overwrite=True,
                       save_config_filename=str(UUID),
                       parser_kwargs={"parser_mode": "omegaconf", "default_config_files": [conf_path]})
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(ckpt_path=cli.trainer.checkpoint_callback.last_model_path, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
