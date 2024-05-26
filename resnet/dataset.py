import torch.utils.data as data
import pytorch_lightning as pl
from base.dataset import GenericDataLoader, SimpleGenericDataset
from typing import Dict, Union
from os import listdir, path
from PIL import Image
from glob import glob
import numpy as np


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
        file = path.basename(file)
        return np.array(Image.open(path.join(self.directory, file)))
            
    
class TargetLoader(GenericDataLoader):
    def __init__(self, directory, n_files=None, files=None):
        self.class_map = {'605': 0, '625': 1}
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
        file = path.basename(file)
        return self.class_map[file[:3]]
    

class GenericDataModule(pl.LightningDataModule):
    def __init__(self, input_loader: GenericDataLoader,
                 target_loader: GenericDataLoader,
                 test_loader: Union[GenericDataLoader, None] = None,
                 batch_size: int = 32, transforms=Dict
                 [str, Dict[str, callable]]):
        super().__init__()
        self.input_loader = input_loader
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.transforms = transforms

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_inp, val_inp = self.input_loader.split()
            train_tar, val_tar = self.target_loader.split()
            self.train = SimpleGenericDataset(
                train_inp, train_tar, self.transforms['train'])
            self.val = SimpleGenericDataset(
                val_inp, val_tar, self.transforms['val'])
        if stage == 'test' or stage is None:
            self.test = SimpleGenericDataset(
                self.input_loader, self.target_loader, self.transforms['val'])

    def train_dataloader(self):
        return data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return data.DataLoader(self.test, batch_size=self.batch_size)
