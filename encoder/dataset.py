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
        file = path.basename(file)  # make sure that the file is just the name
        return np.array(Image.open(path.join(self.directory, file))), torch.tensor(classes.index(file[:3]))


class EncoderDataset(GenericDataset):
    def __init__(self, input_loader, transform=None):
        super(EncoderDataset, self).__init__(
            input_loader, input_loader, transform)

    def unpack(self, inputs, targets):
        inputs, names = inputs
        targets, _ = targets
        return inputs, targets, names


class EncoderDataModule(LightningDataModule):
    def __init__(
            self, input_loader: InputLoader, prediction_loader=None,
            transforms=None, batch_size=32):
        super(EncoderDataModule, self).__init__()
        self.train_data, self.val_data = input_loader.split(
            0.8)  # Split the data
        self.train_loader = InputLoader(
            input_loader.directory, files=self.train_data)
        self.val_loader = InputLoader(
            input_loader.directory, files=self.val_data)
        self.transforms = transforms
        self.batch_size = batch_size
        self.prediction_loader = prediction_loader

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            self.train_set = EncoderDataset(
                self.train_loader, self.transforms['train'])
            self.val_set = EncoderDataset(
                self.val_loader, self.transforms['val'])

        if stage == 'test':
            self.test_set = EncoderDataset(
                self.val_loader, self.transforms['val'])

        if stage == 'predict':
            if self.prediction_loader is not None:
                self.prediction_set = EncoderDataset(
                    self.prediction_loader, self.transforms['val'])
            else:
                self.prediction_set = None

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=7, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=7, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=7, persistent_workers=True)

    def predict_dataloader(self):
        if self.prediction_set is not None:
            return DataLoader(
                self.prediction_set, batch_size=self.batch_size, shuffle=False,
                num_workers=7, persistent_workers=True)
        else:
            warnings.warn('No prediction set provided. Using validation set.')
            return self.val_dataloader()
