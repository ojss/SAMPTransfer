# Offical repository for SAMPTransfer

## Introduction

## Requirements

The code base requires `cuda` to run at its best in terms of speed.
**This codebase uses Pytorch-Lightning and LightningCLI for executing runs**

1. cd into the `SAMPTransfer` folder
2. Create a new conda environment by running: `conda env create -f environment.yml`
3. Install `poetry`: `conda install -c conda-forge poetry`
4. Run `poetry install`

## Running Experiments

### Datasets

1. Make sure all required packages are installed before running the experiments
2. Ensure that there is enough space for datasets to be downloaded - incase they haven't been downloaded already
    1. You can use `download_data.py` to have `torchmeta` download the data for you
    2. In the event the automatic downloader fails, please download the data from links given [**
       here**](https://github.com/tristandeleu/pytorch-meta/blob/c84c8e775f659741f7ad2ab9fbcfc1a78a4e76c9/docs/api_reference/datasets.md)

### Training

1. To start a training run, simply execute `python clr_gat.py -c configs/slurm/mpnn.yml`
    1. You may edit the `.yml` file to reflect your hardware availability
2. This codebase uses Weights and Biases for all the logging needs, you may disable it from the config file if needed

### Evaluation

Even though a testing run is triggered by the completion of training, you may want to test out the pre-trained
checkpoint
artifacts generated during training.

For this purpose, this codebase contains `evaluator.py` to make life easier. To use it modify the following command:

```(bash)
python evaluator.py <dataset> <path_to_checkpoint> <path_to_data> <n_ways> <k_shots> <query_shots> prototune --adapt ot --ft-freeze-backbone
```

We also provide our best pre-trained models in this repo.

## Results

The results on mini-ImageNet and tieredImageNet are given below:

|  |  | mini-ImageNet |  |
| :--- | :---: | :---: | :---: |
| Method (N,K) | Backbone | (5,1) | (5,5) |
| CACTUs-MAML  | Conv4 | 39.90+-0.74 | 53.97+-0.70 |
| CACTUs-Proto  | Conv4 | 39.18+-0.71 | 53.36+-0.70 |
| UMTRA  | Conv4 | 39.93 | 50.73 |
| AAL-ProtoNet  | Conv4 | 37.67+-0.39 | 40.29+-0.68 |
| AAL-MAML++  | Conv4 | 34.57+-0.74 | 49.18+-0.47 |
| UFLST  | Conv4 | 33.77+-0.70 | 45.03+-0.73 |
| ULDA-ProtoNet  | Conv4 | 40.63+-0.61 | 55.41+-0.57 |
| ULDA-MetaNet  | Conv4 | 40.71+-0.62 | 54.49+-0.58 |
| U-SoSN+ArL  | Conv4 | 41.13+-0.84 | 55.39+-0.79 |
| U-MISo  | Conv4 | 41.09 | 55.38 |
| ProtoTransfer  | Conv4 | 45.67+-0.79 | 62.99+-0.75 |
| CUMCA  | Conv4 | 41.12 | 54.55 |
| Meta-GMVAE  | Conv4 | 42.82 | 55.73 |
| Revisiting UML  | Conv4 | 48.12+-0.19 | 65.33+-0.17 |
| CSSL-FSL_Mini64  | Conv4 | 48.53+-1.26 | 63.13+-0.87 |
| $\text{C}^3\text{LR}$  | Conv4 | 47.92+-1.2 | 64.81+-1.15 |
| SAMPTransfer (ours) | Conv4 | 55.75+-0.77 | 67.62+-0.66 |
| SAMPTransfer* (ours) | Conv4b | 61.02+-1.0 | 72.52+-0.68 |
| MAML  | Conv4 | 46.81+-0.77 | 62.13+-0.72 |
| **Supervised** | :---: | :---: | :---: |
| ProtoNet  | Conv4 | 46.44+-0.78 | 66.33+-0.68 |
| MMC  | Conv4 | 50.41+-0.31 | 64.39+-0.24 |
| FEAT  | Conv4 | 55.15 | 71.61 |
| SimpleShot  | Conv4 | 49.69+-0.19 | 66.92+-0.17 |
| Simple CNAPS  | ResNet-18 | 53.2+-0.9 | 70.8+-0.7 |
| Transductive CNAPS  | ResNet-18 | 55.6+-0.9 | 73.1+-0.7 |
| MetaQDA  | Conv4 | 56.41+-0.80 | 72.64+-0.62 |
| Pre+Linear  | Conv4 | 43.87+-0.69 | 63.01+-0.71 |

|  | tieredImageNet |  |
| :--- | :---: | :---: |
| Method (N,K) | (5,1) | (5,5) |
| $\text{C}^3\text{LR}$  | 42.37+-0.77 | 61.77 _+-0.25 |
| ULDA-ProtoNet  | 41.60+-0.64 | 56.28+-0.62 |
| ULDA-MetaOptNet  | 41.77+-0.65 | 56.78+-0.63 |
| U-SoSN+ArL  | 43.68 _+-0.91 | 58.56+-0.74 |
| U-MISo  | 43.01+-0.91 | 57.53+-0.74 |
| SAMPTransfer (ours) | 45.25+-0.89 | 59.75+-0.66 |
| SAMPTransfer* (ours) | 49.10+-0.94 | 65.19+-0.82 |
