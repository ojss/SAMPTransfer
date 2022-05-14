from typing import Tuple

from torch import nn

from graph.gnn_base import GNNReID
from graph.graph_generator import GraphGenerator


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
