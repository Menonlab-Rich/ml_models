from base.dataset import GenericDataLoader, GenericDataset
from os import listdir, path
from PIL import Image
import numpy as np

class TargetLoader(GenericDataLoader):
    def __init__(self, directory):
        self.directory = directory
        files = sorted([f for f in listdir(directory) if f.endswith('.tif')])
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self._read(self.files[idx])

    def __iter__(self):
        for file in self.files:
            yield self._read(file)

    def get_ids(self, i=None, n=None):
        if i is not None and n is not None:
            return self.files[i:i+n]
        elif i is not None:
            return self.files[i]
        return self.files

    def _read(self, file):
        file = path.basename(file)  # make sure that the file is just the name
        return np.array(Image.open(path.join(self.directory, file)))


class InputLoader(GenericDataLoader):
    def __init__(self, directory):
        self.directory = directory
        self.files = sorted(
            [f for f in listdir(directory) if f.endswith('.npz')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self._read(self.files[idx])

    def __iter__(self):
        for c in self.files:
            yield c

    def _read(self, file):
        file = path.basename(file) # make sure that the file is just the name
        return np.load(path.join(self.directory, file))['mask']

    def get_ids(self, i=None):
        if i is not None:
            return self.files[i]
        return self.files

def get_dataset(input_file_path, target_file_path, transform=None):
    input_loader = InputLoader(input_file_path)
    target_loader = TargetLoader(target_file_path)
    return GenericDataset(input_loader, target_loader, transform)