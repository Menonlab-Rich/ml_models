import torch
from torch import nn, cuda
from os import path, getcwd, listdir
from PIL import Image
import albumentations as A
import numpy as np

_cwd = getcwd()
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MULTI_GPU = False
INPUT_PATH = r'D:\CZI_scope\code\preprocess_training\tifs'
# MODEL_PATH = path.join(_cwd, 'model.tar')
MODEL_PATH = r'D:\CZI_scope\code\ml_models\resnet\model.tar'
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_CLASSES = 2
CLASS_MAPPING = {0: '605', 1: '625'}
PREDICTIONS_PATH = path.join(_cwd, 'predictions.png')
NUM_CHANNELS = 1

def INPUT_LOADER():
    files = sorted([path.join(INPUT_PATH, x) for i, x in enumerate(listdir(INPUT_PATH)) if i < 10])
    return [np.array(Image.open(x)) for x in files], files

def TARGET_LOADER():
    files = sorted([path.join(INPUT_PATH, x) for i, x in enumerate(listdir(INPUT_PATH)) if i < 10])
    classes = [1 if x[:3] == CLASS_MAPPING[1] else 0 for x in files]
    return classes, files

INPUT_TRANSFORMS = A.Compose([
    # Add transforms here to preprocess the input data
    A.ToFloat(always_apply=True),
    A.LongestMaxSize(max_size=256, always_apply=True),
    A.PadIfNeeded(256, 256, always_apply=True),
    # Add transforms here to augment the input data for training
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.5)
])

VALIDATION_TRANSFORMS = A.Compose([
    A.ToFloat(always_apply=True),
    A.LongestMaxSize(max_size=256, always_apply=True),
    A.PadIfNeeded(256, 256, always_apply=True)
]) 

TRANSFORMS = {
    'train': {
        'input': lambda x: INPUT_TRANSFORMS(image=x)['image'], # Apply input transforms
        'target': lambda y: torch.tensor(y, dtype=torch.long) # convert to tensor of type long
    },
    'val': {
        'input': lambda x: INPUT_TRANSFORMS(image=x)['image'],
        'target': lambda y: torch.tensor(y, dtype=torch.long) # convert to tensor of type long
    }
}

LOSS_FN = nn.CrossEntropyLoss(label_smoothing=0.1) # Add more parameters as needed

if __name__ == 'config':
    assert not MULTI_GPU or cuda.is_available() and cuda.device_count() > 1, 'Cannot use multiple GPUs'