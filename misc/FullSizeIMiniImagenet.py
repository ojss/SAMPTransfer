# type: ignore[override]
import os
from typing import Any, Callable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pl_bolts.datasets import UnlabeledImagenet
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from dataloaders import get_episode_loader

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib, datasets
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class FullSizeDataset(datasets.ImageFolder):
    def __init__(self, datapath: str, **kwargs):
        super(FullSizeDataset, self).__init__(datapath, **kwargs)


class FullSizeMiniImagenetDataModule(LightningDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/
        Sample-of-Images-from-the-ImageNet-Dataset-used-in-the-ILSVRC-Challenge.png
        :width: 400
        :alt: Imagenet

    Specs:
        - 1000 classes
        - Each image is (3 x varies x varies) (here we default to 3 x 224 x 224)

    Imagenet train, val and test dataloaders.

    The train set is the imagenet train.

    The val set is taken from the train set with `num_imgs_per_val_class` images per class.
    For example if `num_imgs_per_val_class=2` then there will be 2,000 images in the validation set.

    The test set is the official imagenet validation set.

     Example::

        from pl_bolts.datamodules import ImagenetDataModule

        dm = ImagenetDataModule(IMAGENET_PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)
    """

    name = "miniimagenet"

    def __init__(
            self,
            data_dir: str,
            fsl_datapath: str,
            eval_ways: int = 5,
            eval_support_shots: int = 5,
            eval_query_shots: int = 15,
            meta_dir: Optional[str] = None,
            num_imgs_per_val_class: int = 600,
            image_size: int = 224,
            num_workers: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            meta_dir: path to meta.bin file
            num_imgs_per_val_class: how many images per class for the validation set
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use ImageNet dataset loaded from `torchvision` which is not installed yet."
            )

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.fsl_data_path = fsl_datapath
        self.eval_ways = eval_ways
        self.eval_support_shots = eval_support_shots
        self.eval_query_shots = eval_query_shots
        self.num_workers = num_workers
        self.meta_dir = meta_dir
        self.num_imgs_per_val_class = num_imgs_per_val_class
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_samples = 60000 - self.num_imgs_per_val_class * self.num_classes

    @property
    def num_classes(self) -> int:
        """
        Return:

            1000

        """
        return 600

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:
        """This method already assumes you have imagenet2012 downloaded. It validates the data using the meta.bin.

        .. warning:: Please download imagenet on your own first.
        """
        self._verify_splits(self.data_dir, "train")
        self._verify_splits(self.data_dir, "val")

        for split in ["train", "val"]:
            files = os.listdir(os.path.join(self.data_dir, split))

    def train_dataloader(self) -> DataLoader:
        """Uses the train split of imagenet2012 and puts away a portion of it for the validation split."""
        transforms = self.train_transform() if self.train_transforms is None else self.train_transforms

        dataset = FullSizeDataset(
            self.data_dir + "/train",
            transform=transforms,
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        """Uses the part of the train split of imagenet2012  that was not used for training via
        `num_imgs_per_val_class`

        Args:
            batch_size: the batch size
            transforms: the transforms
        """
        dataloader_val = get_episode_loader("miniimagenet", self.fsl_data_path,
                                            ways=self.eval_ways,
                                            shots=self.eval_support_shots,
                                            test_shots=self.eval_query_shots,
                                            batch_size=1,
                                            split='val',
                                            num_workers=2)
        return dataloader_val

    def test_dataloader(self) -> DataLoader:
        """Uses the validation split of imagenet2012 for testing."""
        dataloader_test = get_episode_loader("miniimagenet", self.fsl_data_path,
                                             ways=self.eval_ways,
                                             shots=self.eval_support_shots,
                                             test_shots=self.eval_query_shots,
                                             batch_size=1,
                                             split='test',
                                             shuffle=False,
                                             num_workers=6)
        return dataloader_test

    def train_transform(self) -> Callable:
        """The standard imagenet transforms.

        .. code-block:: python

            transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        preprocessing = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:
        """The standard imagenet transforms for validation.

        .. code-block:: python

            transform_lib.Compose([
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """

        preprocessing = transform_lib.Compose(
            [
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                imagenet_normalization(),
            ]
        )
        return preprocessing
