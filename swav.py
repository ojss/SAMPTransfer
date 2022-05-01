"""Adapted from official swav implementation: https://github.com/facebookresearch/swav."""
import copy
import os
import uuid

import deepspeed
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pl_bolts.models.self_supervised.swav.swav_resnet import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization,
)
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI
from torch import distributed as dist
from torch import nn
from torch.autograd import Variable
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

from feature_extractors.feature_extractor import SWaV_CNN_4Layer
from misc.FullSizeIMiniImagenet import FullSizeMiniImagenetDataModule
from protoclr_obow import Classifier


class SwAV(LightningModule):
    def __init__(
            self,
            gpus: int,
            num_samples: int,
            batch_size: int,
            dataset: str,
            mpnn_opts: dict,
            sup_finetune: bool,
            sup_finetune_lr: float,
            sup_finetune_epochs: int,
            ft_freeze_backbone: bool,
            finetune_batch_norm: bool,
            eval_ways: int = 5,
            num_nodes: int = 1,
            arch: str = "resnet50",
            hidden_mlp: int = 2048,
            feat_dim: int = 128,
            warmup_epochs: int = 10,
            max_epochs: int = 100,
            nmb_prototypes: int = 3000,
            freeze_prototypes_epochs: int = 1,
            temperature: float = 0.1,
            sinkhorn_iterations: int = 3,
            queue_length: int = 0,  # must be divisible by total batch-size
            queue_path: str = "queue",
            epoch_queue_starts: int = 15,
            crops_for_assign: tuple = (0, 1),
            nmb_crops: tuple = (2, 6),
            first_conv: bool = True,
            maxpool1: bool = True,
            optim: str = "adam",
            exclude_bn_bias: bool = False,
            start_lr: float = 0.0,
            learning_rate: float = 1e-3,
            final_lr: float = 0.0,
            weight_decay: float = 1e-6,
            epsilon: float = 0.05,
            **kwargs
    ):
        """
        Args:
            gpus: number of gpus per node used in training, passed to SwAV module
                to manage the queue and select distributed sinkhorn
            num_nodes: number of nodes to train on
            num_samples: number of image samples used for training
            batch_size: batch size per GPU in ddp
            dataset: dataset being used for train/val
            arch: encoder architecture used for pre-training
            hidden_mlp: hidden layer of non-linear projection head, set to 0
                to use a linear projection head
            feat_dim: output dim of the projection head
            warmup_epochs: apply linear warmup for this many epochs
            max_epochs: epoch count for pre-training
            nmb_prototypes: count of prototype vectors
            freeze_prototypes_epochs: epoch till which gradients of prototype layer
                are frozen
            temperature: loss temperature
            sinkhorn_iterations: iterations for sinkhorn normalization
            queue_length: set queue when batch size is small,
                must be divisible by total batch-size (i.e. total_gpus * batch_size),
                set to 0 to remove the queue
            queue_path: folder within the logs directory
            epoch_queue_starts: start uing the queue after this epoch
            crops_for_assign: list of crop ids for computing assignment
            nmb_crops: number of global and local crops, ex: [2, 6]
            first_conv: keep first conv same as the original resnet architecture,
                if set to false it is replace by a kernel 3, stride 1 conv (cifar-10)
            maxpool1: keep first maxpool layer same as the original resnet architecture,
                if set to false, first maxpool is turned off (cifar10, maybe stl10)
            optim: optimizer to use
            exclude_bn_bias: exclude batchnorm and bias layers from weight decay in optimizers
            start_lr: starting lr for linear warmup
            learning_rate: learning rate
            final_lr: float = final learning rate for cosine weight decay
            weight_decay: weight decay for optimizer
            epsilon: epsilon val for swav assignments
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.nmb_prototypes = nmb_prototypes
        self.freeze_prototypes_epochs = freeze_prototypes_epochs
        self.sinkhorn_iterations = sinkhorn_iterations

        self.queue_length = queue_length
        self.queue_path = queue_path
        self.epoch_queue_starts = epoch_queue_starts
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops

        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optim
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.mpnn_opts = mpnn_opts

        # fine-tuning
        self.sup_finetune = sup_finetune
        self.sup_finetune_lr = sup_finetune_lr
        self.sup_finetune_epochs = sup_finetune_epochs
        self.ft_freeze_backbone = ft_freeze_backbone
        self.finetune_batch_norm = finetune_batch_norm
        self.eval_ways = eval_ways

        if self.gpus * self.num_nodes > 1:
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

        self.model = self.init_model()

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        self.queue = None
        self.softmax = nn.Softmax(dim=1)

    def setup(self, stage):
        if self.queue_length > 0:
            queue_folder = os.path.join(self.logger.log_dir, self.queue_path)
            if not os.path.exists(queue_folder):
                os.makedirs(queue_folder)

            self.queue_path = os.path.join(queue_folder, "queue" + str(self.trainer.global_rank) + ".pth")

            if os.path.isfile(self.queue_path):
                self.queue = torch.load(self.queue_path)["queue"]

    def init_model(self):
        if self.arch == "conv4":
            backbone = SWaV_CNN_4Layer
            args = dict(
                in_channels=3,
                out_channels=64, hidden_size=64,
                global_pooling=True,
                eval_mode=False,
                last_maxpool=False,
                ada_maxpool=False,
                normalize=True,
                output_dim=self.feat_dim,
                hidden_mlp=self.hidden_mlp,
                nmb_prototypes=self.nmb_prototypes,
                mpnn_opts=self.mpnn_opts
            )
        elif self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50

        if self.arch in ["resnet18", "resnet50"]:
            args = dict(normalize=True,
                        hidden_mlp=self.hidden_mlp,
                        output_dim=self.feat_dim,
                        nmb_prototypes=self.nmb_prototypes,
                        first_conv=self.first_conv,
                        maxpool1=self.maxpool1, )

        return backbone(
            **args
        )

    def forward(self, x):
        # pass single batch from the resnet backbone
        return self.model.forward_backbone(x)

    def on_train_epoch_start(self):
        if self.queue_length > 0:
            if self.trainer.current_epoch >= self.epoch_queue_starts and self.queue is None:
                self.queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.queue_length // self.gpus,  # change to nodes * gpus once multi-node
                    self.feat_dim,
                )

            if self.queue is not None:
                self.queue = self.queue.to(self.device)

        self.use_the_queue = False

    def on_train_epoch_end(self) -> None:
        if self.queue is not None:
            torch.save({"queue": self.queue}, self.queue_path)

    def on_after_backward(self):
        if self.current_epoch < self.freeze_prototypes_epochs:
            for name, p in self.model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

    def shared_step(self, batch):
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        inputs, y = batch
        inputs = inputs[:-1]  # remove online train/eval transforms at this point

        # 1. normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        embedding, output = self.model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # 3. swav loss computation
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)]

                # 4. time to use the queue
                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(self.queue[i], self.model.prototypes.weight.t()), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # 5. get assignments
                q = torch.exp(out / self.epsilon).t()
                q = self.get_assignments(q, self.sinkhorn_iterations)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                p = self.softmax(output[bs * v: bs * (v + 1)] / self.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

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
        z_a_i = encoder.forward_backbone(x_a_i, eval_mode=True).flatten(1)
        encoder.eval()
        if self.mpnn_opts["_use"]:
            self.eval()
            if self.mpnn_opts["task_adapt"]:
                _, _, z = encoder.forward_gat(torch.cat([x_a_i, x_b_i]))
                z = z[0]
                z_a_i = z[:support_size, :]
            else:
                _, _, z_a_i = self.mpnn_shared_step(x_a_i, y_a_i)
                z_a_i = z_a_i[0]
            self.train()
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
                    _, _, output = encoder.forward_gat(z_batch, y_batch)
                    output = output[0]
                else:
                    output = encoder.forward_backbone(z_batch, eval_mode=True).flatten(1)

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
                _, _, output = encoder.forward_gat(torch.cat([x_a_i, x_b_i]), )
                output = output[0][support_size:, :]
            else:
                _, _, output = encoder.forward_gat(x_b_i)
                output = output[0]
        else:
            output = encoder.forward_backbone(x_b_i, eval_mode=True).flatten(1)

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

        if self.sup_finetune:
            loss, acc = self.supervised_finetuning(self.model,
                                                   episode=batch,
                                                   inner_lr=self.sup_finetune_lr,
                                                   total_epoch=self.sup_finetune_epochs,
                                                   freeze_backbone=self.ft_freeze_backbone,
                                                   finetune_batch_norm=self.finetune_batch_norm,
                                                   device=self.device,
                                                   n_way=self.eval_ways)
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

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{"params": params, "weight_decay": weight_decay}, {"params": excluded_params, "weight_decay": 0.0}]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = deepspeed.ops.adam.FusedAdam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @staticmethod
    def add_model_specific_args(parser):

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")

        parser.add_argument(
            "--nmb_crops", type=int, default=[2, 4], nargs="+", help="list of number of crops (example: [2, 6])"
        )
        parser.add_argument(
            "--size_crops", type=int, default=[96, 36], nargs="+", help="crops resolutions (example: [224, 96])"
        )
        parser.add_argument(
            "--min_scale_crops",
            type=float,
            default=[0.33, 0.10],
            nargs="+",
            help="argument in RandomResizedCrop (example: [0.14, 0.05])",
        )
        parser.add_argument(
            "--max_scale_crops",
            type=float,
            default=[1, 0.33],
            nargs="+",
            help="argument in RandomResizedCrop (example: [1., 0.14])",
        )

        # swav params
        parser.add_argument(
            "--crops_for_assign",
            type=int,
            nargs="+",
            default=[0, 1],
            help="list of crops id used for computing assignments",
        )

        parser.add_argument("--input_height", default=224, type=int)

        return parser


class SWaVCli(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--job_name", type=str, default="swav_local")
        parser.link_arguments("data.eval_ways", "model.eval_ways")
        parser = SwAV.add_model_specific_args(parser)
        # training params
        parser.set_defaults({"trainer.fast_dev_run": 1, "trainer.num_nodes": 1, "trainer.gpus": 1})


def cli_main():
    from pl_bolts.models.self_supervised.swav.transforms import SwAVEvalDataTransform, SwAVTrainDataTransform
    UUID = uuid.uuid4()
    OmegaConf.register_resolver("uuid", lambda: UUID)

    cli = SWaVCli(SwAV, FullSizeMiniImagenetDataModule, run=False, save_config_filename=str(UUID),
                  save_config_overwrite=True,
                  parser_kwargs={"parser_mode": "omegaconf"})
    args = cli.config["model"]
    # parser = ArgumentParser()
    #
    # # model args
    # parser = SwAV.add_model_specific_args(parser)
    # args = parser.parse_args()

    if args["dataset"] in ("imagenet", "miniimagenet"):
        normalization = imagenet_normalization()
        # args["nmb_prototypes"] = 3000
        dm = cli.datamodule
        cli.model.num_samples = dm.num_samples
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SwAVTrainDataTransform(
        normalize=normalization,
        size_crops=cli.config["size_crops"],
        nmb_crops=cli.config["nmb_crops"],
        min_scale_crops=cli.config["min_scale_crops"],
        max_scale_crops=cli.config["max_scale_crops"],
        gaussian_blur=cli.config["gaussian_blur"],
        jitter_strength=cli.config["jitter_strength"],
    )

    dm.val_transforms = SwAVEvalDataTransform(
        normalize=normalization,
        size_crops=cli.config["size_crops"],
        nmb_crops=cli.config["nmb_crops"],
        min_scale_crops=cli.config["min_scale_crops"],
        max_scale_crops=cli.config["max_scale_crops"],
        gaussian_blur=cli.config["gaussian_blur"],
        jitter_strength=cli.config["jitter_strength"],
    )

    # swav model init
    model = cli.model

    # trainer = Trainer(
    #     max_epochs=args.max_epochs,
    #     max_steps=None if args.max_steps == -1 else args.max_steps,
    #     gpus=args.gpus,
    #     num_nodes=args.num_nodes,
    #     accelerator="ddp" if args.gpus > 1 else None,
    #     sync_batchnorm=True if args.gpus > 1 else False,
    #     precision=32 if args.fp32 else 16,
    #     callbacks=callbacks,
    #     fast_dev_run=args.fast_dev_run,
    # )

    cli.trainer.fit(model, datamodule=dm)
    cli.trainer.test(ckpt_path=cli.trainer.checkpoint_callback.best_model_path, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
