import os

import pytest
import sys

sys.path.insert(1, os.path.abspath("../"))

from cli import custom_cli
from protoclr_obow import PCLROBoW
from dataloaders import UnlabelledDataModule
from bow.bow_extractor import BoWExtractor, BoWExtractorMultipleLevels


def test_clr_on_bow():
    c = custom_cli.MyCLI(PCLROBoW, UnlabelledDataModule, run=False,
                         save_config_overwrite=True,
                         parser_kwargs={"parser_mode": "omegaconf",
                                        "default_config_files": ["configs/test_clr_on_bow.yml"]})
    with pytest.raises(AttributeError):
        c.model.feature_extractor_teacher
    with pytest.raises(AttributeError):
        c.model.bow_predictor
    assert isinstance(c.model.bow_extractor, BoWExtractorMultipleLevels)
    c.trainer.fit(c.model, c.datamodule)
    return 1
