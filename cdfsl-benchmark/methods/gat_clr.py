# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
from backbone import GATResNet10
import utils
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch import nn
from .gnn_base import GNNReID
from .graph_generator import GraphGenerator


class GATCLR(nn.Module):
    """Calculate the UMTRA-style loss on a batch of images.
    If shots=1 and only two views are served for each image,
    this corresponds exactly to UMTRA except that it uses ProtoNets
    instead of MAML.

    Parameters:
        - model_func: The encoder network.
        - shots: The number of support shots.
    """

    def __init__(self, model_func, shots=1):
        super(GATCLR, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.feature = model_func()
        self.top1 = utils.AverageMeter()
        self.shots = shots
        self.loss_cnn = True
        self.emb_dim = self.backbone(torch.randn(1, 3, 224, 224)).shape[-1]
        self.mpnn_opts = {
            "_use": True,
            "loss_cnn": True,
            "scaling_ce": 1,
            "adapt": "ot",
            "temperature": 0.2,
            "output_train_gnn": "plain",
            "graph_params": {
                "sim_type": "correlation",
                "thresh": "no",
                "set_negative": "hard"},
            "gnn_params": {
                "pretrained_path": "no",
                "red": 1,
                "cat": 0,
                "every": 0,
                "gnn": {
                    "num_layers": 1,
                    "aggregator": "add",
                    "num_heads": 2,
                    "attention": "dot",
                    "mlp": 1,
                    "dropout_mlp": 0.1,
                    "norm1": 1,
                    "norm2": 1,
                    "res1": 1,
                    "res2": 1,
                    "dropout_1": 0.1,
                    "dropout_2": 0.1,
                    "mult_attr": 0},
                "classifier": {
                    "neck": 0,
                    "num_classes": 0,
                    "dropout_p": 0.4,
                    "use_batchnorm": 0}
            }
        }
        self.gnn = GNNReID(self.mpnn_dev, self.mpnn_opts["gnn_params"], self.emb_dim)
        self.graph_generator = GraphGenerator(self.mpnn_dev, **self.mpnn_opts["graph_params"])
        self.final_feat_dim = self.backbone.final_feat_dim

    def get_scores(self, z, ways, shots):
        z_support = z[:ways * shots]
        z_query = z[ways * shots:]

        # Get prototypes
        z_proto = z_support.view(ways, shots, -1).mean(1)  # the shape of z is [n_data, n_dim]

        # Calculate loss and accuracies
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def forward(self, x):
        # Treat the first dim as way, the second as shots
        ways = x.size(0)
        n_views = x.size(1)
        shots = self.shots
        query_shots = n_views - shots
        x_support = x[:, :shots].reshape((ways * shots, *x.shape[-3:]))
        x_support = Variable(x_support.cuda())
        x_query = x[:, shots:].reshape((ways * query_shots, *x.shape[-3:]))
        x_query = Variable(x_query.cuda())

        # Create dummy query labels
        y_query = torch.arange(ways).unsqueeze(1)  # shot dim
        y_query = y_query.repeat(1, query_shots)
        y_query = y_query.view(-1).cuda()

        # Extract features
        x_both = torch.cat([x_support, x_query], 0)
        if isinstance(self.feature, GATResNet10):
            z_cnn, z = self.feature(x_both)
        else:
            z = self.feature(x_both)

        scores = self.get_scores(z, ways, shots)
        if self.mpnn_opts["_use"]:
            z = z.flatten(1)
            edge_attr, edge_index, z = self.graph_generator.get_graph(z)
            _, (z,) = self.gnn(z, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])

        # GAT bits
        gat_scores = self.get_scores(z, ways, shots)

        return scores, gat_scores, y_query

    def forward_loss(self, x):
        loss = 0.
        scores, gat_scores, y = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y).cpu().sum()
        self.top1.update(correct.item() * 100 / (y.size(0) + 0.0), y.size(0))
        if self.loss_cnn:
            loss = self.loss_fn(scores, y)
        loss += self.loss_fn(gat_scores, y)
        return loss

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        for i, x in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i,
                                                                                                        len(train_loader),
                                                                                                        avg_loss / float(
                                                                                                            i + 1),
                                                                                                        self.top1.val,
                                                                                                        self.top1.avg))


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
