import torch
from torch import nn, cuda
from os import path, getcwd, listdir
from PIL import Image
import albumentations as A
import numpy as np
from base.dataset import GenericDataLoader
from base.config import BaseConfigHandler
import toml


class Config(BaseConfigHandler):
    def __init__(self):

        INPUT_TRANSFORMS = A.Compose([
            # Add transforms here to preprocess the input data
            A.ToFloat(always_apply=True),
            # Add transforms here to augment the input data for training
            A.RandomBrightnessContrast(p=.5),
            A.GaussNoise(var_limit=(1e-4, 1e-3), p=.5),
        ])

        VAL_TRANSFORMS = A.Compose([
            A.ToFloat(always_apply=True),
        ])

        self.config = {
            'transform': {
                'train': {
                    'input': lambda x: INPUT_TRANSFORMS(image=x)['image'],
                    'target': lambda y: torch.tensor(y, dtype=torch.long)
                },
                'val': {
                    'input': lambda x: VAL_TRANSFORMS(image=x)['image'],
                    'target': lambda y: torch.tensor(y, dtype=torch.long)
                }
            },
        }

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def save(self, path: str):
        with open(path, 'w') as f:
            toml.dump(self.config, f)

    def load(self, path: str):
        toml_dict = toml.load(path)
        # extend self.config with the toml_dict
        self.config.update(toml_dict)
    
    def get(self, key, default=None):
        return self.config.get(key, default)




class InputLoader(GenericDataLoader):
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

    def get_ids(self, i=None):
        if i is not None:
            return self.files[i]
        return self.files

    def _read(self, file):
        file = path.basename(file)  # make sure that the file is just the name
        return np.array(Image.open(path.join(self.directory, file)))


class TargetLoader(GenericDataLoader):
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

config = Config()
config.load(path.join(path.dirname(__file__), 'config.toml'))
_input_loader = InputLoader(config['directories']['inputs'])
_target_loader = TargetLoader(config['directories']['targets'])

config['input_loader'] = _input_loader
config['target_loader'] = _target_loader
config['loss_fn'] = nn.CrossEntropyLoss(label_smoothing=0.1)

if __name__ == 'config':
    if config['directories']['create']:
        from os import makedirs
        makedirs(config['directories']['data'], exist_ok=True)
        makedirs(config['directories']['results'], exist_ok=True)
        print("Directories created")