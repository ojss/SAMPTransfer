import json
import os

import h5py
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms, datasets
from misc import dino_utils

from dataloaders import get_transforms, get_omniglot_transform, get_custom_transform, identity_transform, \
    get_episode_loader


class UnlabelledDataset(Dataset):
    def __init__(self, dataset, datapath, split, transform=None, tfm_method=None,
                 n_support=1, n_query=1, n_images=None, n_classes=None,
                 seed=10, no_aug_support=False, no_aug_query=False,
                 img_size_orig=(224, 224), img_size_crop=(84, 84)):
        """
        Args:
            dataset (string): Dataset name.
            datapath (string): Directory containing the datasets.
            split (string): The dataset split to load.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            n_support (int): Number of support examples
            n_query (int): Number of query examples
            no_aug_support (bool): Wheteher to not apply any augmentations to the support
            no_aug_query (bool): Wheteher to not apply any augmentations to the query
            n_images (int): Limit the number of images to load.
            n_classes (int): Limit the number of classes to load.
            seed (int): Random seed to for selecting images to load.
        """
        self.n_support = n_support
        self.n_query = n_query
        self.img_size_crop = (28, 28) if dataset == 'omniglot' else img_size_crop
        self.img_size_orig = (28, 28) if dataset == 'omniglot' else img_size_orig
        self.no_aug_support = no_aug_support
        self.no_aug_query = no_aug_query

        # Get the data or paths
        self.dataset = dataset
        self.data = self._extract_data_from_hdf5(dataset, datapath, split,
                                                 n_classes, seed)

        # Optionally only load a subset of images
        if n_images is not None:
            random_idxs = np.random.RandomState(seed).permutation(len(self))[:n_images]
            self.data = self.data[random_idxs]

        # Get transform
        if transform is not None:
            self.transform = transform
        else:
            if tfm_method is not None:
                self.transform = get_transforms(tfm_method, self.img_size_crop)
                self.original_transform = identity_transform(self.img_size_orig)
            # if self.dataset == 'cub':
            #     self.transform = transforms.Compose([
            #         get_cub_default_transform(self.img_size_crop),
            #         get_custom_transform(self.img_size_crop)])
            #     self.original_transform = transforms.Compose([
            #         get_cub_default_transform(self.img_size_crop),
            #         transforms.ToTensor()])
            elif self.dataset == 'omniglot':
                self.transform = get_omniglot_transform((28, 28))
                self.original_transform = identity_transform((28, 28))
            else:
                self.transform = transforms.RandomResizedCrop(self.img_size_orig, scale=(.5, 1.))
                self.original_transform = identity_transform(self.img_size_orig)

    def _extract_data_from_hdf5(self, dataset, datapath, split,
                                n_classes, seed):
        datapath = os.path.join(datapath, dataset)

        # Load omniglot
        if dataset == 'omniglot':
            classes = []
            with h5py.File(os.path.join(datapath, 'data.hdf5'), 'r') as f_data:
                with open(os.path.join(datapath,
                                       'vinyals_{}_labels.json'.format(split))) as f_labels:
                    labels = json.load(f_labels)
                    for label in labels:
                        img_set, alphabet, character = label
                        classes.append(f_data[img_set][alphabet][character][()])
        # Load mini-imageNet
        else:
            with h5py.File(os.path.join(datapath, split + '_data.hdf5'), 'r') as f:
                datasets = f['datasets']
                classes = [datasets[k][()] for k in datasets.keys()]

        # Optionally filter out some classes
        if n_classes is not None:
            random_idxs = np.random.RandomState(seed).permutation(len(classes))[:n_classes]
            classes = [classes[i] for i in random_idxs]

        # Collect in single array
        data = np.concatenate(classes)
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.n_query == 1:
            img = Image.fromarray(self.data[index])
            img1 = self.original_transform(img)
            img2 = self.transform(img)
            if self.data[index].shape[0] == 1:
                img1 = torch.cat([img1] * 3)
                img2 = torch.cat([img2] * 3)
            return img1, img2


class FullSizeDataset(datasets.ImageFolder):
    def __init__(self, datapath: str, local_crops_number: int = 0):
        super(FullSizeDataset, self).__init__(datapath)
        self.augs = DataAugmentationDINO((0.5, 1.0), (0.05, 0.4), local_crops_number)

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        image = self.loader(path)
        img1, img2 = self.augs(image)
        return img1, img2


class FewShotDatamodule(pl.LightningDataModule):
    def __init__(self, dataset, datapath, tfm_method=None, full_size_path=None,
                 n_support=1, n_query=1, n_images=None, n_classes=None, batch_size=50, num_workers=8,
                 no_aug_support=False, no_aug_query=False, merge_train_val=True,
                 eval_ways=5, eval_support_shots=5, eval_query_shots=15, img_size_orig: tuple = (84, 84),
                 img_size_crop: tuple = (84, 84), local_crops_number: int = 0, **kwargs):
        super().__init__()

        self.n_images = n_images
        self.n_support = n_support
        self.n_query = n_query
        self.n_classes = n_classes
        self.img_size_orig = (28, 28) if dataset == "omniglot" else img_size_orig
        self.img_size = (28, 28) if dataset == 'omniglot' else img_size_crop
        self.no_aug_support = no_aug_support
        self.no_aug_query = no_aug_query
        self.tfm_method = tfm_method

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Get the data or paths
        self.dataset = dataset
        self.datapath = datapath
        self.full_size_path = full_size_path
        self.local_crops_number = local_crops_number

        self.eval_ways = eval_ways
        self.eval_support_shots = eval_support_shots
        self.eval_query_shots = eval_query_shots

        self.merge_train_val = merge_train_val

        self.kwargs = kwargs

    def setup(self, stage=None):
        if self.img_size_orig == (224, 224):
            self.dataset_train = FullSizeDataset(self.full_size_path, self.local_crops_number)
        else:
            self.dataset_train = UnlabelledDataset(self.dataset,
                                                   self.datapath, split='train',
                                                   transform=None,
                                                   tfm_method=self.tfm_method,
                                                   n_images=self.n_images,
                                                   n_classes=self.n_classes,
                                                   n_support=self.n_support,
                                                   n_query=self.n_query,
                                                   no_aug_support=self.no_aug_support,
                                                   no_aug_query=self.no_aug_query,
                                                   img_size_crop=self.img_size, img_size_orig=self.img_size_orig)

            if self.merge_train_val:
                dataset_val = UnlabelledDataset(self.dataset, self.datapath, 'val',
                                                transform=None,
                                                tfm_method=self.tfm_method,
                                                n_support=self.n_support,
                                                n_query=self.n_query,
                                                no_aug_support=self.no_aug_support,
                                                no_aug_query=self.no_aug_query,
                                                img_size_crop=self.img_size, img_size_orig=self.img_size_orig)
                self.dataset_train = ConcatDataset([self.dataset_train, dataset_val])

    def train_dataloader(self):
        dataloader_train = DataLoader(self.dataset_train,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=torch.cuda.is_available())
        return dataloader_train

    def val_dataloader(self):
        dataloader_val = get_episode_loader(self.dataset, self.datapath,
                                            ways=self.eval_ways,
                                            shots=self.eval_support_shots,
                                            test_shots=self.eval_query_shots,
                                            batch_size=1,
                                            split='val',
                                            **self.kwargs)
        return dataloader_val

    def test_dataloader(self):
        dataloader_test = get_episode_loader(self.dataset, self.datapath,
                                             ways=self.eval_ways,
                                             shots=self.eval_support_shots,
                                             test_shots=self.eval_query_shots,
                                             batch_size=1,
                                             split='test',
                                             shuffle=False,
                                             num_workers=2,
                                             **self.kwargs)
        return dataloader_test


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(0.1),
            dino_utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
