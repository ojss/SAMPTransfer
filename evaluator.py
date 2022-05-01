import argparse

import pytorch_lightning as pl

from feature_extractors.feature_extractor import CNN_4Layer
from callbacks import ConfidenceIntervalCallback
from dataloaders import UnlabelledDataModule
from protoclr_obow import PCLROBoW


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str)
    parser.add_argument("--ev_ways", default=5, type=int)
    parser.add_argument("--ev_supp_shots", default=5, type=int)
    parser.add_argument("--ev_query_shots", default=15, type=int)
    parser.add_argument("--sup_finetune", action='store_true')
    args = parser.parse_args()
    kwargs = dict(n_support=1, n_query=3, batch_size=64, lr_decay_step=25000, lr_decay_rate=0.5,
                  sup_finetune=args.sup_finetune,
                  feature_extractor=CNN_4Layer(3, 64, 64), bow_levels=['block4'],
                  bow_extractor_opts={'inv_delta': 15, 'num_words': 8192}, bow_predictor_opts={'kappa': 5})
    model = PCLROBoW(**kwargs)
    # xs = torch.load("ckpts/model_net_checkpoint_335.pth.tar", map_location='cpu')
    # model.load_state_dict(xs['network'])
    model.load_from_checkpoint(args.c, **kwargs)
    datamodule = UnlabelledDataModule(dataset='miniimagenet',
                                      datapath='/home/nfs/oshirekar/unsupervised_ml/data/',
                                      split='test',
                                      img_size_orig=(84, 84),
                                      img_size_crop=(60, 60), eval_ways=args.ev_ways,
                                      eval_support_shots=args.ev_supp_shots, eval_query_shots=args.ev_query_shots)
    trainer = pl.Trainer(
        gpus=-1,
        limit_test_batches=600,
        callbacks=[ConfidenceIntervalCallback(log_to_wb=False)]
    )
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cli_main()
