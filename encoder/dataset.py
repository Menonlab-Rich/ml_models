from os import listdir, path
from PIL import Image
from glob import glob
from base.dataset import GenericDataset, SimpleGenericDataset, GenericDataLoader
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import warnings
 

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
        return np.array(Image.open(path.join(self.directory, file))), file

    def split(self, proportions=[0.8, 0.2]):
        '''
        Split the files into n proportions
        '''
        assert sum(proportions) == 1, 'Proportions should sum to 1'

        # get the number of files for each proportion
        n_files = [int(p * len(self)) for p in proportions]

        files = []
        all_files = self.files.copy()
        # sample the files for each proportion and remove them from the list to avoid duplicates
        for n in n_files:
            sampled_files = np.random.choice(all_files, n, replace=False)
            # remove the sampled files
            all_files = np.setdiff1d(all_files, sampled_files)
            files.append(sampled_files)

        return [InputLoader(self.directory, files=f) for f in files]


class EncoderDataset(SimpleGenericDataset):
    def __init__(self, input_loader, transform=None):
        super(EncoderDataset, self).__init__(
            input_loader, input_loader, transform)
    
    def unpack(self, inputs, targets):
        inputs, names = inputs
        targets, _ = targets
        return inputs, targets, names


class EncoderDataModule(LightningDataModule):
    def __init__(self, input_loader: InputLoader, prediction_loader=None, transforms=None, batch_size=32):
        super(EncoderDataModule, self).__init__()
        self.train_loader, self.val_loader = input_loader.split() # Split the data
        self.transforms = transforms
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            self.train_set = EncoderDataset(self.train_loader, self.transforms['train'])
            self.val_set = EncoderDataset(self.val_loader, self.transforms['val'])
        
        if stage == 'test':
            self.test_set = EncoderDataset(self.val_loader, self.transforms['val'])
        
        if stage == 'predict':
            if self.prediction_loader is not None:
                self.prediction_set = EncoderDataset(self.prediction_loader, self.transforms['val'])
            else:
                self.prediction_set = None
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False)
    
    def predict_dataloader(self):
        if self.prediction_set is not None:
            return DataLoader(
                self.prediction_set, batch_size=self.batch_size, shuffle=False)
        else:
            warnings.warn('No prediction set provided. Using validation set.')
            return self.val_dataloader()
    
        
        
        

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False)
