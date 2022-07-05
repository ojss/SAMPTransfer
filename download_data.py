from torchmeta.datasets.helpers import cub, omniglot, miniimagenet, tieredimagenet, fc100, cifar_fs
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                           '../'))
import configs


# Parse command line / default arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='omniglot',
                    help='Dataset to load')
parser.add_argument('--datapath', type=str, default=configs.data_path,
                    help='Path, where dataset are stored')
args = parser.parse_args()

# Select dataset
dataset = args.dataset
if dataset == 'omniglot':
    dataset_func = omniglot
elif dataset == 'miniimagenet':
    dataset_func = miniimagenet
elif dataset == 'cub':
    dataset_func = cub
elif dataset == 'tieredimagenet':
    dataset_func = tieredimagenet
elif dataset == 'cifar_fs':
    dataset_func = cifar_fs
elif dataset == 'fc100':
    dataset_func = fc100
else:
    raise ValueError("No such dataset available. Please choose from\
                     ['omniglot', 'miniimagenet', 'tieredimagenet',\
                      'cub, cifar_fs, doublemnist, triplemnist']")
print(dataset_func)
# Create dataset for labelled images
dataset = dataset_func(args.datapath,
                       ways=5,
                       shots=1,
                       meta_train=True,
                       download=True)
