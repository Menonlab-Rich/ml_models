from os import listdir, path
from PIL import Image
from glob import glob
from base.dataset import GenericDataset, GenericDataLoader
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import warnings
from config import Config, CONFIG_FILE_PATH
import torch

config = Config(CONFIG_FILE_PATH)
classes = list(config.weights.keys())


class InputLoader(GenericDataLoader):
    def __init__(self, directory, n_files=None, files=None):
        '''
        Create a loader for the input files
        Input files are the images that will be used as input to the model

        Parameters:
            directory (str): The directory containing the input files
            n_files (int): The number of files to sample
            files (List[str]): The list of files to use (optional. Files must be in the directory)
        '''
        self.directory = directory
        # Check if files are provided
        if files is not None:
            self.files = files
        # If files are not provided, check if directory is actually a directory (not a glob pattern)
        elif not path.isdir(directory):
            # Could be a glob pattern
            self.files = sorted(glob(directory))
            self.directory = path.dirname(directory)
        # If files is not provided and directory is a directory, list the files
        else:
            self.files = sorted(
                [f for f in listdir(directory) if f.endswith('.tif')])
        if n_files is not None:
            # Sample n_files at random
            self.files = np.random.choice(self.files, n_files, replace=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self._read(self.files[idx])

    def __iter__(self):
        for file in self.files:
            yield self._read(file)

    def get_ids(self, i=None, batch_size=1):
        if i is not None:
            # return the file name(s) for the index and batch size
            return self.files[i:i+batch_size - 1]
        return self.files

    def _read(self, file):
        '''
        Return the file as a numpy array
        '''
        file = path.basename(file)  # make sure that the file is just the name
        return np.array(Image.open(path.join(self.directory, file)))

    def post_split(self, train_ids, val_ids):
        return InputLoader(
            self.directory, files=train_ids), InputLoader(
            self.directory, files=val_ids)


class TargetLoader(GenericDataLoader):
    def __init__(self, directory, n_files=None, files=None):
        '''
        Create a loader for the target files
        The target files are the class labels for the input files

        Parameters:
            directory (str): The directory containing the target files
            n_files (int): The number of files to sample
            files (List[str]): The list of files to use (optional. Files must be in the directory)
        '''
        self.directory = directory
        # Check if files are provided
        if files is not None:
            self.files = files
        # If files are not provided, check if directory is actually a directory (not a glob pattern)
        elif not path.isdir(directory):
            # Could be a glob pattern
            self.files = sorted(glob(directory))
            self.directory = path.dirname(directory)
        # If files is not provided and directory is a directory, list the files
        else:
            self.files = sorted(
                [f for f in listdir(directory) if f.endswith('.tif')])
        if n_files is not None:
            # Sample n_files at random
            self.files = np.random.choice(self.files, n_files, replace=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self._read(self.files[idx])

    def __iter__(self):
        for file in self.files:
            yield self._read(file)

    def get_ids(self, i=None, batch_size=1):
        if i is not None:
            # return the file name(s) for the index and batch size
            return self.files[i:i+batch_size - 1]
        return self.files

    def _read(self, file: str) -> int:
        '''
        Return the class of the file based on the first three characters of the file name
        '''
        file = path.basename(file)  # make sure that the file is just the name
        return classes.index(file[:3])

    def post_split(self, train_ids, val_ids):
        return TargetLoader(
            self.directory, files=train_ids), TargetLoader(
            self.directory, files=val_ids)


class ResnetDataset(GenericDataset):
    def __init__(self, input_loader, target_loader, transform=None):
        super(ResnetDataset, self).__init__(
            input_loader, target_loader, transform)


class ResnetDataModule(LightningDataModule):
    def __init__(
            self, input_loader: InputLoader, target_loader: TargetLoader,
            prediction_loader=None, transforms=None, batch_size=32, n_workers=7):
        super(ResnetDataModule, self).__init__()
        self.train_inputs, self.val_inputs = input_loader.split(
            0.8)  # Split the data
        self.train_targets, self.val_targets = target_loader.split(
            0.8)  # Split the data
        self.transforms = transforms
        self.batch_size = batch_size
        self.prediction_loader = prediction_loader
        self.n_workers = n_workers

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            self.train_set = ResnetDataset(
                self.train_inputs, self.train_targets, self.transforms)
            self.val_set = ResnetDataset(
                self.val_inputs, self.val_targets, self.transforms)

        if stage == 'test':
            self.test_set = ResnetDataset(
                self.val_inputs, self.val_targets, self.transforms)

        if stage == 'predict':
            if self.prediction_loader is not None:
                self.prediction_set = ResnetDataset(
                    self.prediction_loader, self.transforms)
            else:
                self.prediction_set = None

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_workers, persistent_workers=True)

    def predict_dataloader(self):
        if self.prediction_set is not None:
            return DataLoader(
                self.prediction_set, batch_size=self.batch_size, shuffle=False,
                num_workers=self.n_workers, persistent_workers=True)
        else:
            warnings.warn('No prediction set provided. Using validation set.')
            return self.val_dataloader()
