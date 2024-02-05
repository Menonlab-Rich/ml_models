import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

# Do not change these parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Change these parameters to customize your model
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 8 # based on number of cores
IMAGE_SIZE = 256 # size of your input images, you can increase it if you have large memory
CHANNELS_INPUT = 1 # Grayscale
CHANNELS_OUTPUT = 3 # RGB
LOSS_FN = lambda x, y: nn.L1Loss()(x, y) * 100 # L1 loss with a weight of 100
NUM_EPOCHS = 6
LOAD_MODEL = True # set to True if you want to load a pre-trained model
SAVE_MODEL = True # set to True to save the model
CHECKPOINT = "unet.pth.tar" # Saved modle filename
TRAIN_IMG_DIR="/home/rich/Documents/school/menon/ml_models/unet/data/landscapes/gray/*.jpg" # input images path
TARGET_DIR="/home/rich/Documents/school/menon/ml_models/unet/data/landscapes/color/*.jpg" # target images path 
EXAMPLES_DIR="/home/rich/Documents/school/menon/ml_models/unet/results" # Where to save example images



# Augmentation pipeline
# Find documentation here: https://albumentations.ai/docs/
transform_both = A.Compose(
    [
        # A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        # A.Normalize(
        #     mean=[0., 0., 0.], std=[1., 1., 1.], max_pixel_value=255.0,
        # ),
        # A.Rotate(limit=35, p=0.5),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),
        
        # Keep ToTensor as the last transformation
        # It converts the numpy images to torch tensors
        # Normalizing the images to be in the range [0, 1]
        # and transposing the channels from HWC to CHW format
        ToTensorV2(),
    ],
    additional_targets={'target': 'image'} # required if target is an image. Could also be set to mask, or other supported key
)

transform_input = A.Compose(
    [
        # add noise to the input image
        A.GaussNoise(p=0.5),
    ]
)


