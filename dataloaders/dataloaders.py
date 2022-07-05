__all__ = ['collate_task', 'collate_task_batch', 'get_episode_loader', 'UnlabelledDataset', 'get_cub_default_transform',
           'get_simCLR_transform', 'get_omniglot_transform', 'get_custom_transform', 'identity_transform',
           'UnlabelledDataModule']

import io
import json
import os
from collections import OrderedDict

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageFilter, ImageOps
from pl_bolts.transforms.self_supervised import RandomTranslateWithReflect
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
from torchmeta.datasets.helpers import (omniglot, miniimagenet, tieredimagenet,
                                        cub, cifar_fs, doublemnist, triplemnist)
from torchmeta.utils.data import MetaDataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

from . import miniImageNet_few_shot


def collate_task(task):
    if isinstance(task, Dataset):
        return default_collate([task[idx] for idx in range(len(task))])
    elif isinstance(task, OrderedDict):
        return OrderedDict([(key, collate_task(subtask))
                            for (key, subtask) in task.items()])
    else:
        raise NotImplementedError()


def collate_task_batch(batch):
    return default_collate([collate_task(task) for task in batch])


def get_episode_loader(dataset, datapath, ways, shots, test_shots, batch_size,
                       split, download=True, shuffle=True, num_workers=0, **kwargs):
    """Create an episode data loader for a torchmeta dataset. Can also
    include unlabelled data for semi-supervised learning.

    dataset: String. Name of the dataset to use.
    datapath: String. Path, where dataset are stored.
    ways: Integer. Number of ways N.
    shots: Integer. Number of shots K for support set.
    test_shots: Integer. Number of images in query set.
    batch_size: Integer. Number of tasks per iteration.
    split: String. One of ['train', 'val', 'test']
    download: Boolean. Whether to download the data.
    shuffle: Boolean. Whether to shuffle episodes.
    """
    # Select dataset
    if dataset == 'omniglot':
        dataset_func = omniglot
    elif dataset == 'miniimagenet':
        dataset_func = miniimagenet
    elif dataset == 'tieredimagenet':
        dataset_func = tieredimagenet
    elif dataset == 'cub':
        dataset_func = cub
    elif dataset == 'cifar_fs':
        dataset_func = cifar_fs
    elif dataset == 'doublemnist':
        dataset_func = doublemnist
    elif dataset == 'triplemnist':
        dataset_func = triplemnist
    else:
        raise ValueError("No such dataset available. Please choose from\
                         ['omniglot', 'miniimagenet', 'tieredimagenet',\
                          'cub, cifar_fs, doublemnist, triplemnist']")

    # Collect arguments that are the same for all possible sub-datasets
    kwargs = {'download': download,
              'meta_train': split == 'train',
              'meta_val': split == 'val',
              'meta_test': split == 'test',
              'shuffle': shuffle}

    # Create dataset for labelled images
    dataset_name = dataset
    dataset = dataset_func(datapath,
                           ways=ways,
                           shots=shots,
                           test_shots=test_shots,
                           **kwargs)

    print('Supervised data loader for {}:{}.'.format(dataset_name, split))
    # Standard supervised meta-learning dataloader
    collate_fn = collate_task_batch if batch_size else collate_task
    return MetaDataLoader(dataset, batch_size=batch_size,
                          collate_fn=collate_fn,
                          num_workers=num_workers,
                          pin_memory=torch.cuda.is_available())


class ULDS(ImageFolder):
    def __init__(self, datapath, split, transform=None,
                 n_support=1, n_query=1, n_images=None, n_classes=None,
                 no_aug_support=False, no_aug_query=False, img_size_orig=(224, 224), img_size_crop=(84, 84)):
        super(ULDS, self).__init__(datapath + f"/{split}")
        self.n_support = n_support
        self.n_query = n_query
        self.split = split
        self.n_images = n_images
        self.n_classes = n_classes
        self.no_aug_support = no_aug_support
        self.no_aug_query = no_aug_query
        self.img_size_orig = img_size_orig
        self.img_size_crop = img_size_crop
        if transform is not None:
            self.transform = get_transforms(transform, self.img_size_crop)
        else:
            self.transform = get_custom_transform(self.img_size_crop)
        self.original_transform = identity_transform(self.img_size_orig)

    def __getitem__(self, idx):
        path, _ = self.samples[idx]
        image = self.loader(path)
        view_list = []
        originals = []
        for _ in range(self.n_support):
            if not self.no_aug_support:
                originals.append(self.transform(image).unsqueeze(0))
            else:
                assert self.n_support == 1
                originals.append(self.original_transform(image).unsqueeze(0))

        for _ in range(self.n_query):
            if not self.no_aug_query:
                view_list.append(self.transform(image).unsqueeze(0))
            else:
                assert self.n_query == 1
                view_list.append(self.original_transform(image).unsqueeze(0))

        return dict(origs=torch.cat(originals), views=torch.cat(view_list))


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
            elif self.dataset == 'cub':
                self.transform = transforms.Compose([
                    get_cub_default_transform(self.img_size_crop),
                    get_custom_transform(self.img_size_crop)])
                self.original_transform = transforms.Compose([
                    get_cub_default_transform(self.img_size_crop),
                    transforms.ToTensor()])
            elif self.dataset == 'omniglot':
                self.transform = get_omniglot_transform((28, 28))
                self.original_transform = identity_transform((28, 28))
            else:
                self.transform = get_custom_transform(self.img_size_crop)
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
                class_names = list(datasets.keys())
                classes = [datasets[k][()] for k in datasets.keys()]
                labels = [np.repeat([i], len(datasets[k][()])) for i, k in enumerate(class_names)]
                labels = np.array(labels).flatten()
            if self.dataset == "miniimagenet":
                self.targets = LabelEncoder().fit_transform(labels)

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
        if self.dataset == 'cub' or self.dataset == "tieredimagenet":
            image = Image.open(io.BytesIO(self.data[index])).convert('RGB')
        else:
            image = Image.fromarray(self.data[index])
        if self.dataset == "miniimagenet":
            target = self.targets[index]

        view_list = []
        originals = []
        for _ in range(self.n_support):
            if not self.no_aug_support:
                originals.append(self.transform(image).unsqueeze(0))
            else:
                assert self.n_support == 1
                originals.append(self.original_transform(image).unsqueeze(0))

        for _ in range(self.n_query):
            if not self.no_aug_query:
                view_list.append(self.transform(image).unsqueeze(0))
            else:
                assert self.n_query == 1
                view_list.append(self.original_transform(image).unsqueeze(0))

        return dict(origs=torch.cat(originals), views=torch.cat(view_list),
                    labels=np.repeat(target, self.n_support + self.n_query) if self.dataset == "miniimagenet" else 0)


def get_cub_default_transform(size):
    return transforms.Compose([
        transforms.Resize([int(size[0] * 1.5), int(size[1] * 1.5)]),
        transforms.CenterCrop(size)])


def get_simCLR_transform(img_shape):
    """Adapted from https://github.com/sthalles/SimCLR/blob/master/data_aug/dataset_wrapper.py"""
    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                          saturation=0.8, hue=0.2)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=img_shape[-2:]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          # GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                          transforms.ToTensor()])
    return data_transforms


def get_omniglot_transform(img_shape):
    data_transforms = transforms.Compose([
        transforms.Resize(img_shape[-2:]),
        transforms.RandomResizedCrop(size=img_shape[-2:],
                                     scale=(0.6, 1.4)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        #   transforms.Lambda(lambda t: F.dropout(t, p=0.3)),
        transforms.RandomErasing()
    ])
    return data_transforms


def get_custom_transform(img_shape):
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                          saturation=0.4, hue=0.1)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=img_shape[-2:],
                                                                       scale=(0.5, 1.0)),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.ToTensor()])
    return data_transforms


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def get_transforms(method, img_size):
    if method == "vicreg":
        return vicreg_transforms(img_size)
    elif method == "amdim":
        return AMDIM_transforms(img_size)


def vicreg_transforms(img_size):
    tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=img_size[-2:], scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return tfms


def AMDIM_transforms(image_size):
    transforms_list = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.25),
        transforms.ToTensor(),
    ])
    return transforms_list


def identity_transform(img_shape):
    return transforms.Compose([transforms.Resize(img_shape),
                               transforms.ToTensor()])


class UnlabelledDataModule(pl.LightningDataModule):
    def __init__(self, dataset, datapath, split, transform=None, tfm_method=None, full_size_path=None,
                 n_support=1, n_query=1, n_images=None, n_classes=None, batch_size=50, num_workers=8,
                 seed=10, no_aug_support=False, no_aug_query=False, merge_train_val=True, mode='val',
                 eval_ways=5, eval_support_shots=5, eval_query_shots=15, img_size_orig=(224, 224),
                 img_size_crop=(84, 84), use_folder=False, **kwargs):
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

        self.use_folder = use_folder

        self.mode = mode

        self.eval_ways = eval_ways
        self.eval_support_shots = eval_support_shots
        self.eval_query_shots = eval_query_shots

        self.merge_train_val = merge_train_val

        self.kwargs = kwargs

    def setup(self, stage=None):
        if self.img_size_orig == [224, 224]:
            self.dataset_train = ULDS(self.full_size_path, split='train', transform=None, n_images=self.n_images,
                                      n_classes=self.n_classes, n_support=self.n_support, n_query=self.n_query,
                                      no_aug_query=self.no_aug_query, no_aug_support=self.no_aug_support,
                                      img_size_crop=self.img_size,
                                      img_size_orig=self.img_size_orig)
        elif self.use_folder and self.img_size_orig == [84, 84]:
            self.dataset_train = ULDS(self.full_size_path, split='train', transform=self.tfm_method,
                                      n_images=self.n_images,
                                      n_classes=self.n_classes, n_support=self.n_support, n_query=self.n_query,
                                      no_aug_query=self.no_aug_query, no_aug_support=self.no_aug_support,
                                      img_size_crop=self.img_size,
                                      img_size_orig=self.img_size_orig)
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
        if self.merge_train_val and not self.use_folder:
            dataset_val = UnlabelledDataset(self.dataset, self.datapath, 'val',
                                            transform=None,
                                            tfm_method=self.tfm_method,
                                            n_support=self.n_support,
                                            n_query=self.n_query,
                                            no_aug_support=self.no_aug_support,
                                            no_aug_query=self.no_aug_query,
                                            img_size_crop=self.img_size, img_size_orig=self.img_size_orig)
        elif self.merge_train_val and self.use_folder and self.img_size_orig == [84, 84]:
            dataset_val = ULDS(self.full_size_path, split='val', transform=self.tfm_method, n_images=self.n_images,
                               n_classes=self.n_classes, n_support=self.n_support, n_query=self.n_query,
                               no_aug_query=self.no_aug_query, no_aug_support=self.no_aug_support,
                               img_size_crop=self.img_size,
                               img_size_orig=self.img_size_orig)
        if self.merge_train_val:
            self.dataset_train = ConcatDataset([self.dataset_train, dataset_val])
        else:
            self.dataset_train = self.dataset_train

    def train_dataloader(self):
        dataloader_train = DataLoader(self.dataset_train,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=torch.cuda.is_available())
        return dataloader_train

    def val_dataloader(self):
        if self.img_size_orig == [224, 224]:
            datamgr = miniImageNet_few_shot.SetDataManager(self.img_size_orig[0], n_way=self.eval_ways,
                                                           n_support=self.eval_support_shots,
                                                           n_query=self.eval_query_shots, split='val', n_eposide=15,
                                                           miniImageNet_path=self.full_size_path + "/val")
            dataloader_val = datamgr.get_data_loader(aug=False)
        else:
            dataloader_val = get_episode_loader(self.dataset, self.datapath,
                                                ways=self.eval_ways,
                                                shots=self.eval_support_shots,
                                                test_shots=self.eval_query_shots,
                                                batch_size=1,
                                                split='val',
                                                shuffle=False,
                                                **self.kwargs)
        return dataloader_val

    def test_dataloader(self):
        if self.img_size_orig == [224, 224]:
            datamgr = miniImageNet_few_shot.SetDataManager(self.img_size_orig[0], n_way=self.eval_ways,
                                                           n_support=self.eval_support_shots, n_eposide=1000,
                                                           n_query=self.eval_query_shots, split='test',
                                                           miniImageNet_path=self.full_size_path + "/test")
            dataloader_test = datamgr.get_data_loader(aug=False)
        else:
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
