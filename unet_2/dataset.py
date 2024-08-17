from base.dataset import GenericDataLoader, Transformer, GenericDataset
from pytorch_lightning import LightningDataModule
import torch
import tifffile as tiff
import numpy as np
import os
from os import path, listdir
from glob import glob
import pickle
import warnings
from torch.utils.data import DataLoader
from itertools import islice
from functools import cache
from types import GeneratorType


class InputLoader(GenericDataLoader):
    """
    Loader for input files (images) to be used as input to the model.
    """

    def __init__(self, directory, n_files=None, files=None, patch_size=None, step_size=None):
        self.directory = directory
        self.patch_size = patch_size
        self.step_size = step_size
        if files is not None:
            self.files = files
        elif not path.isdir(directory):
            self.files = sorted(glob(directory))
            self.directory = path.dirname(directory)
        else:
            self.files = sorted([f for f in listdir(directory) if f.endswith('.tif')])
        if n_files is not None:
            self.files = self.files[:n_files]

        first_image = tiff.memmap(path.join(self.directory, self.files[0]))
        self.img_height, self.img_width = first_image.shape[:2]

    @property
    @cache
    def _patches_per_image(self):
        return ((self.img_height - self.patch_size) // self.step_size + 1) * \
               ((self.img_width - self.patch_size) // self.step_size + 1) \
                   if self.patch_size and self.step_size else 1

    @property
    def _y_steps(self):
        return range(0, self.img_height - self.patch_size + 1, self.step_size) \
            if self.patch_size and self.step_size else [0]

    @property
    def _x_steps(self):
        return range(0, self.img_width - self.patch_size + 1, self.step_size) \
            if self.patch_size and self.step_size else [0]

    @property
    @cache
    def _y_remainder(self):
        return (self.img_height - self.patch_size) % self.step_size \
            if self.patch_size and self.step_size else 0

    @property
    @cache
    def _x_remainder(self):
        return (self.img_width - self.patch_size) % self.step_size \
            if self.patch_size and self.step_size else 0

    @cache
    def __len__(self):
        return len(self.files) * self._patches_per_image

    def __getitem__(self, idx):
        file_idx = idx // self._patches_per_image
        patch_idx = idx % self._patches_per_image
        return self._read(self.files[file_idx], patch_idx)

    def __iter__(self):
        for file in self.files:
            yield self._read(file)

    def get_ids(self, i=None, **_):
        file_idx = i // self._patches_per_image
        patch_idx = i % self._patches_per_image
        return f'{self.files[file_idx]}_{patch_idx}'

    def _generate_patches(self, img):
        if not self.patch_size or not self.step_size:
            yield img
            return

        for y in self._y_steps:
            for x in self._x_steps:
                yield img[y:y+self.patch_size, x:x+self.patch_size]
        if self._y_remainder:
            for x in range(0, img.shape[1] - self.patch_size + 1, self.step_size):
                yield img[-self.patch_size:, x:x+self.patch_size]
        if self._x_remainder:
            for y in range(0, img.shape[0] - self.patch_size + 1, self.step_size):
                yield img[y:y+self.patch_size, -self.patch_size:]
        if self._y_remainder and self._x_remainder:
            yield img[-self.patch_size:, -self.patch_size:]

    def _read(self, file, patch_idx=None):
        img = tiff.memmap(path.join(self.directory, path.basename(file)))
        if self.patch_size and self.step_size:
            generator = self._generate_patches(img)
            if patch_idx is not None:
                return next(islice(generator, patch_idx, None))
            return generator
        return img, file

    def post_split(self, train_ids, val_ids):
        return InputLoader(self.directory, files=train_ids), InputLoader(self.directory, files=val_ids)

    def collate_fn(self, batch):
        patches, filenames = zip(*batch)
        patches = [torch.from_numpy(patch) if not torch.is_tensor(patch) else patch for patch in patches]
        return torch.stack(patches), filenames


class TargetLoader(GenericDataLoader):
    """
    Loader for target files (masks) to be used as input to the model.
    """

    def __init__(self, directory, n_files=None, files=None, patch_size=None, step_size=None):
        self.directory = directory
        self.patch_size = patch_size
        self.step_size = step_size
        if files is not None:
            self.files = files
        elif not path.isdir(directory):
            self.files = sorted(glob(directory))
            self.directory = path.dirname(directory)
        else:
            self.files = sorted([f for f in listdir(directory) if f.endswith('.tif')])
        if n_files is not None:
            self.files = self.files[:n_files]

        first_image = tiff.memmap(path.join(self.directory, self.files[0]))
        self.img_height, self.img_width = first_image.shape[:2]

    @property
    @cache
    def _patches_per_image(self):
        return ((self.img_height - self.patch_size) // self.step_size + 1) * \
               ((self.img_width - self.patch_size) // self.step_size + 1) \
                   if self.patch_size and self.step_size else 1

    @property
    def _y_steps(self):
        return range(0, self.img_height - self.patch_size + 1, self.step_size) \
            if self.patch_size and self.step_size else [0]

    @property
    def _x_steps(self):
        return range(0, self.img_width - self.patch_size + 1, self.step_size) \
            if self.patch_size and self.step_size else [0]

    @property
    @cache
    def _y_remainder(self):
        return (self.img_height - self.patch_size) % self.step_size \
            if self.patch_size and self.step_size else 0

    @property
    @cache
    def _x_remainder(self):
        return (self.img_width - self.patch_size) % self.step_size \
            if self.patch_size and self.step_size else 0

    @cache
    def __len__(self):
        return len(self.files) * self._patches_per_image

    def __getitem__(self, idx):
        file_idx = idx // self._patches_per_image
        patch_idx = idx % self._patches_per_image
        return self._read(self.files[file_idx], patch_idx)

    def __iter__(self):
        for file in self.files:
            yield self._read(file)

    def get_ids(self, i=None, **_):
        file_idx = i // self._patches_per_image
        patch_idx = i % self._patches_per_image
        return f'{self.files[file_idx]}_{patch_idx}'

    def _generate_patches(self, img):
        if not self.patch_size or not self.step_size:
            yield img
            return

        for y in self._y_steps:
            for x in self._x_steps:
                yield img[y:y+self.patch_size, x:x+self.patch_size]
        if self._y_remainder:
            for x in range(0, img.shape[1] - self.patch_size + 1, self.step_size):
                yield img[-self.patch_size:, x:x+self.patch_size]
        if self._x_remainder:
            for y in range(0, img.shape[0] - self.patch_size + 1, self.step_size):
                yield img[y:y+self.patch_size, -self.patch_size:]
        if self._y_remainder and self._x_remainder:
            yield img[-self.patch_size:, -self.patch_size:]

    def _read(self, file, patch_idx=None):
        img = np.load(path.join(self.directory, path.basename(file)))['mask'] # Load the mask from the .npz file
        if self.patch_size and self.step_size:
            generator = self._generate_patches(img)
            if patch_idx is not None:
                return next(islice(generator, patch_idx, None))
            return generator
        return img, file

    def post_split(self, train_ids, val_ids):
        return InputLoader(self.directory, files=train_ids), InputLoader(self.directory, files=val_ids)

    def collate_fn(self, batch):
        patches, filenames = zip(*batch)
        patches = [torch.from_numpy(patch) if not torch.is_tensor(patch) else patch for patch in patches]
        return torch.stack(patches), filenames


class UnetDataset(GenericDataset):
    """
    Dataset class for ResNet model.
    """

    def __init__(self, input_loader, target_loader, transform=None):
        """
        Initialize the ResnetDataset.

        Parameters:
        input_loader (InputLoader): Loader for input files.
        target_loader (TargetLoader): Loader for target files.
        transform (callable): Transformation function (default: None).
        """
        super(UnetDataset, self).__init__(
            input_loader, target_loader, transform)

    def unpack(self, inputs, targets):
        inputs, filenames = inputs
        # If the input is a generator, convert it to a list
        if isinstance(inputs, GeneratorType):
            inputs = list(inputs)
        return inputs, targets, filenames
    
    def collate_fn(self, batch):
        inputs, targets, filenames = zip(*batch)
        
        # Collate the inputs and targets
        inputs = self.input_loader.collate_fn(inputs)
        targets = self.target_loader.collate_fn(targets)
        
        return inputs, targets, filenames

class UNetDataModule(LightningDataModule):
    """
    PyTorch Lightning Data Module for ResNet model.
    """

    def __init__(
            self, input_loader: InputLoader = None,
            target_loader: TargetLoader = None, prediction_loader=None,
            test_loaders=None, transforms=None, batch_size=32, n_workers=7,
            split_ratio=0.8, no_split=False):
        """
        Initialize the ResnetDataModule.

        Parameters:
        input_loader (InputLoader): Loader for input files.
        target_loader (TargetLoader): Loader for target files.
        prediction_loaders (list): List of loaders for prediction (default: None).
        test_loaders (list): List of loaders for testing (default: None).
        transforms (callable): Transformation function (default: None).
        batch_size (int): Batch size (default: 32).
        n_workers (int): Number of workers (default: 7).
        no_split (bool): If True, do not split the data (default: False).
        """
        super().__init__()
        self.input_loader = self._load_if_bytes(input_loader)
        self.target_loader = self._load_if_bytes(target_loader)
        self.test_loaders = test_loaders
        self.transforms = transforms
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.prediction_loader = prediction_loader
        if input_loader is None or target_loader is None:
            # If input_loader or target_loader is not provided, we are probably loading from a state dict
            return
        if not no_split:
            self.train_inputs, self.val_inputs = self._split_data(
                self.input_loader, split_ratio)
            self.train_targets, self.val_targets = self._split_data(
                self.target_loader, split_ratio)
        else:
            self.train_inputs = self.val_inputs = self.input_loader
            self.train_targets = self.val_targets = self.target_loader

        self.save_hyperparameters({
            "batch_size": batch_size, "n_workers": n_workers,
        })

    def _split_data(self, loader: GenericDataLoader, split_ratio: float):
        """
        Split the data into training and validation sets.

        Parameters:
        data (list): List of data.
        split_ratio (float): Ratio to split the data.

        Returns:
        Tuple[list, list]: Training and validation sets.
        """
        return loader.split(split_ratio)

    def _load_if_bytes(self, obj):
        if isinstance(obj, bytes) and len(obj) > 0:
            return pickle.loads(obj)
        elif isinstance(obj, bytes):
            return None
        return obj

    def setup(self, stage: str):
        """
        Set up the data module for different stages of training.

        Parameters:
        stage (str): Stage of training ('fit', 'test', or 'predict').
        """
        if stage in ('fit', None):
            self.train_set = UnetDataset(
                self.train_inputs, self.train_targets, self.transforms)
            self.val_set = UnetDataset(
                self.val_inputs, self.val_targets, self.transforms)

        if stage == 'test':
            self.test_set = self._get_test_set()

        if stage == 'predict':
            self.prediction_set = UnetDataset(
                self.prediction_loader, None, self.transforms)

    def _get_test_set(self):
        if isinstance(self.test_loaders, str):
            if self.test_loaders.lower() == 'validation':
                if getattr(self, 'val_inputs', None) is None:
                    warnings.warn(
                        'Validation set not found. Trying to load from input_loader state dict.')
                    try:
                        self.val_inputs = self.val_set.input_loader
                        self.val_targets = self.val_set.target_loader
                    except AttributeError:
                        raise ValueError('Validation set not found.')
                return UnetDataset(
                    self.val_inputs, self.val_targets, self.transforms)
        elif self.test_loaders:
            return UnetDataset(*self.test_loaders, self.transforms)
        return None

    def __setattr__(self, name: str, value: any) -> None:
        """
        Set an attribute of the data module.

        Parameters:
        name (str): Name of the attribute.
        value (Any): Value to set.
        """

        # Prevent overwriting existing attributes
        # Required when loading from state dict
        if name in (
            'input_loader', 'target_loader', 'prediction_loader',
                'test_loaders'):
            if getattr(self, name, None) is not None:
                warnings.warn(f'Refusing to set {name} as it is already set.')
                return
        super().__setattr__(name, value)

    def state_dict(self):
        """
        Save the state of the data module.

        Returns:
        dict: State dictionary.
        """
        return {
            'batch_size': self.batch_size,
            'n_workers': self.n_workers,
            'input_loader': pickle.dumps(self.input_loader),
            'target_loader': pickle.dumps(self.target_loader),
            'prediction_loader': pickle.dumps(self.prediction_loader) if self.prediction_loader else b'',
            'test_loaders': pickle.dumps(self.test_loaders) if self.test_loaders else b'',
            'transforms': pickle.dumps(self.transforms) if self.transforms else b'',
        }

    def load_state_dict(self, state_dict):
        """
        Load the state of the data module.

        Parameters:
        state_dict (dict): State dictionary.
        """
        try:
            self.batch_size = state_dict['batch_size']
            self.n_workers = state_dict['n_workers']
            self.input_loader = self._load_if_bytes(state_dict['input_loader'])
            self.target_loader = self._load_if_bytes(
                state_dict['target_loader'])
            self.prediction_loader = self._load_if_bytes(
                state_dict['prediction_loader'])
            self.test_loaders = self._load_if_bytes(state_dict['test_loaders'])
            self.transforms = self._load_if_bytes(state_dict['transforms'])
            if (not self.input_loader or not self.target_loader) and (not self.prediction_loader and not self.test_loaders):
                raise ValueError('Input or target loader not found.')
        except KeyError as e:
            warnings.warn(f'Error in loading state dict: {e}')
        self.save_hyperparameters({
            "batch_size": self.batch_size, "n_workers": self.n_workers,
        })
        self.train_inputs, self.val_inputs = self._split_data(
            self.input_loader, 0.8)
        self.train_targets, self.val_targets = self._split_data(
            self.target_loader, 0.8)

    def train_dataloader(self):
        """
        Get the DataLoader for training data.

        Returns:
        DataLoader: DataLoader for training data.
        """
        return self._create_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        """
        Get the DataLoader for validation data.

        Returns:
        DataLoader: DataLoader for validation data.
        """
        return self._create_dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        """
        Get the DataLoader for test data.

        Returns:
        DataLoader: DataLoader for test data, or None if test set is not defined.
        """
        return self._create_dataloader(
            self.test_set, shuffle=False) if self.test_set else None

    def predict_dataloader(self):
        """
        Get the DataLoader for prediction data.

        Returns:
        DataLoader: DataLoader for prediction data, or validation DataLoader if prediction set is not defined.
        """
        if self.prediction_set:
            return self._create_dataloader(self.prediction_set, shuffle=False)
        warnings.warn('No prediction set provided. Using validation set.')
        return self.val_dataloader()

    def _create_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.n_workers, persistent_workers=True)
