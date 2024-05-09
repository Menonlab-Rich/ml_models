import torch
from torch import nn, cuda
from os import path, getcwd, listdir
from PIL import Image
import albumentations as A
import numpy as np
from dataset import GenericDataLoader
import cv2

_cwd = getcwd()
# ROOT_DIR = path.dirname(_cwd)
ROOT_DIR = r'D:\CZI_scope\code\ml_models\resnet'
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MULTI_GPU = False
INPUT_PATH = r'D:\CZI_scope\code\preprocess_training\validation'
# MODEL_PATH = path.join(_cwd, 'model.tar')
MODEL_PATH = r'D:\CZI_scope\code\ml_models\resnet\model.tar'
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_CLASSES = 2
CLASS_MAPPING = {0: '605', 1: '625'}
PREDICTIONS_PATH = path.join(_cwd, 'predictions.png')
NUM_CHANNELS = 1
# DST_SAVE_DIR = path.join(_cwd, 'data')
DST_SAVE_DIR = r'D:\CZI_scope\code\ml_models\resnet\data'
REPORT_PATH = path.join(_cwd, 'logs', 'resnet_prediction_report.txt')


class InputLoader(GenericDataLoader):
    def __init__(self, directory):
        self.directory = directory
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


class TargetLoader(GenericDataLoader):
    def __init__(self, directory):
        files = sorted([f for f in listdir(directory) if f.endswith('.tif')])
        self.classes = [1 if x[:3] == CLASS_MAPPING[0] else 1 for x in files]

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

VAL_TRANSFORMS = A.Compose(
    [A.ToFloat(always_apply=True),
     A.LongestMaxSize(max_size=256, always_apply=True),
     A.PadIfNeeded(
         256, 256, always_apply=True, border_mode=cv2.BORDER_CONSTANT,
        value=0),])


TRANSFORMS = {
    'train': {
        # Apply input transforms
        'input': lambda x: INPUT_TRANSFORMS(image=x)['image'],
        # convert to tensor of type long
        'target': lambda y: torch.tensor(y, dtype=torch.long)
    },
    'val': {
        'input': lambda x: VAL_TRANSFORMS(image=x)['image'],
        # convert to tensor of type long
        'target': lambda y: torch.tensor(y, dtype=torch.long)
    }
}

LOSS_FN = nn.BCEWithLogitsLoss()

if __name__ == 'config':
    assert not MULTI_GPU or cuda.is_available(
    ) and cuda.device_count() > 1, 'Cannot use multiple GPUs'
