from os import listdir, path
from PIL import Image
from glob import glob
from base.dataset import GenericDataset, GenericDataLoader
import numpy as np
import torch

class InputLoader(GenericDataLoader):
    def __init__(self, directory):
        self.directory = directory
        # Check if the path is actually a directory
        if not path.isdir(directory):
            # Could be a glob pattern
            self.files = sorted(glob(directory))
            self.directory = path.dirname(directory)
        else:
            self.files = sorted(
                [f for f in listdir(directory) if f.endswith('.tif')])

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
        return np.array(Image.open(path.join(self.directory, file)))
    


class EncoderDataset(GenericDataset):
    def __init__(self, input_loader, transform=None):
        super(EncoderDataset, self).__init__(input_loader, input_loader, transform)
