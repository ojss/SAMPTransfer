import torch
import argparse

from bow.feature_extractor import CNN_4Layer
from protoclr_obow import PCLROBoW
from dataloaders import UnlabelledDataModule
import pytorch_lightning as pl


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str)
    args = parser.parse_args()
    model = PCLROBoW.load_from_checkpoint(args.c,
                                          map_location=torch.device('cpu'), n_support=1, n_query=3, batch_size=64,
                                          lr_decay_step=25000, lr_decay_rate=0.5,
                                          feature_extractor=CNN_4Layer(3, 64, 64), bow_levels=['block4'],
                                          bow_extractor_opts={'inv_delta': 15, 'num_words': 8192},
                                          bow_predictor_opts={'kappa': 5})
    datamodule = UnlabelledDataModule(dataset='miniimagenet',
                                      datapath='/home/ojass/projects/unsupervised-meta-learning/data/untarred/miniimagenet',
                                      split='test',
                                      img_size_orig=(84, 84),
                                      img_size_crop=(60, 60))
    trainer = pl.Trainer(
        # gpus=-1,
        limit_test_batches=600,
    )
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cli_main()
