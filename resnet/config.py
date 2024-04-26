import torch
from torch import nn, cuda
from os import path, getcwd, listdir
from PIL import Image
import albumentations as A
import numpy as np
from glob import glob
from dataset import GenericDataLoader

_cwd = getcwd()
ROOT_DIR = path.dirname(_cwd)
#ROOT_DIR = r'D:\CZI_scope\code\ml_models\resnet'
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MULTI_GPU = False
INPUT_PATH = '/scratch/general/nfs1/u0977428/transfer/preprocess/tifs/'
MODEL_PATH = path.join(_cwd, 'model.tar')
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_CLASSES = 2
CLASS_MAPPING = {0: '605', 1: '625'}
PREDICTIONS_PATH = path.join(_cwd, 'predictions.png')
NUM_CHANNELS = 1
DST_SAVE_DIR = path.join(_cwd, 'data')
#DST_SAVE_DIR = r'D:\CZI_scope\code\ml_models\resnet\data'

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
        file = path.basename(file) # make sure that the file is just the name
        return np.array(Image.open(path.join(self.directory, file)))
    
class TargetLoader(GenericDataLoader):
    def __init__(self, directory):
        files = sorted([f for f in listdir(directory) if f.endswith('.tif')])
        self.classes = [1 if x[:3] == CLASS_MAPPING[1] else 0 for x in files]

    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, idx):
        return self.classes[idx]
    
    def __iter__(self):
        for c in self.classes:
            yield c
    
    def get_ids(self, i=None):
        if i is not None:
            return self.classes[i]
        return self.classes
    
INPUT_LOADER = InputLoader(INPUT_PATH)
TARGET_LOADER = TargetLoader(INPUT_PATH)
        

INPUT_TRANSFORMS = A.Compose([
    # Add transforms here to preprocess the input data
    A.ToFloat(always_apply=True),
    A.LongestMaxSize(max_size=256, always_apply=True),
    A.PadIfNeeded(256, 256, always_apply=True),
    # Add transforms here to augment the input data for training
   A.RandomBrightnessContrast(p=.5),
   A.GaussNoise(var_limit=(1e-4, 1e-3), p=.5),
])

VAL_TRANSFORMS = ([
    A.ToFloat(always_apply=True),
    A.LongestMaxSize(max_size=256, always_apply=True),
    A.PadIfNeeded(256, 256, always_apply=True),
])

 

TRANSFORMS = {
    'train': {
        'input': lambda x: INPUT_TRANSFORMS(image=x)['image'], # Apply input transforms
        'target': lambda y: torch.tensor(y, dtype=torch.long) # convert to tensor of type long
    },
    'val': {
        'input': lambda x: VAL_TRANSFORMS(image=x)['image'],
        'target': lambda y: torch.tensor(y, dtype=torch.long) # convert to tensor of type long
    }
}

LOSS_FN = nn.CrossEntropyLoss(label_smoothing=0.1) # Add more parameters as needed

if __name__ == 'config':
    assert not MULTI_GPU or cuda.is_available() and cuda.device_count() > 1, 'Cannot use multiple GPUs'
