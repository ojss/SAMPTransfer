import os

import pytest
import sys

sys.path.insert(1, os.path.abspath("../"))

from cli import custom_cli
from protoclr_obow import PCLROBoW
from dataloaders import UnlabelledDataModule

import torch_geometric.nn as gnn


def test_edge():
    c = custom_cli.MyCLI(PCLROBoW, UnlabelledDataModule, run=False,
                         save_config_overwrite=True,
                         parser_kwargs={"parser_mode": "omegaconf",
                                        "default_config_files": ["configs/test_edge.yml"]})
    assert isinstance(c.model.ec1, gnn.DynamicEdgeConv)
    with pytest.raises(AttributeError):
        c.model.feature_extractor_teacher
    with pytest.raises(AttributeError):
        c.model.bow_predictor
    with pytest.raises(AttributeError):
        c.model.bow_extractor
    c.trainer.fit(c.model, c.datamodule)
