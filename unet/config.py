import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

# Do not change these parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Change these parameters to customize your model
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 8  # based on number of cores
IMAGE_SIZE = 256  # size of your input images, you can increase it if you have large memory
CHANNELS_INPUT = 1  # Grayscale
CHANNELS_OUTPUT = 3  # RGB


def LOSS_FN(x, y): return nn.L1Loss()(
    x, y) * 100  # L1 loss with a weight of 100


NUM_EPOCHS = 6
LOAD_MODEL = False  # set to True if you want to load a pre-trained model
SAVE_MODEL = True  # set to True to save the model
CHECKPOINT = "unet.pth.tar"  # Saved modle filename
# input images path
TRAIN_IMG_DIR = "/home/rich/Documents/school/menon/ml_models/unet/data/landscapes/gray/*.jpg"
# target images path
TARGET_DIR = "/home/rich/Documents/school/menon/ml_models/unet/data/landscapes/color/*.jpg"
# Where to save example images
EXAMPLES_DIR = "/home/rich/Documents/school/menon/ml_models/unet/results"
# Colormap for input images. Change gray to another colormap if you want to use a different colormap
# Leave else None to correctly handle 3 channel images
CMAP = 'gray' if CHANNELS_INPUT == 1 else None


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
    # required if target is an image. Could also be set to mask, or other supported key
    additional_targets={'target': 'image'}
)

# You can add additional transformations to the input images if you want
# Just make sure not to add ToTensorV2() to the input transformations
# This is because ToTensorV2() should be the last transformation and it should be applied to both the input and target images
transform_input = A.Compose(
    [
        # add noise to the input image
        A.GaussNoise(p=0.5),
    ]
)
# You can add transformations to the target images if you want by following the same pattern.
# Just make sure not to add ToTensorV2() to the input transformations
# This is because ToTensorV2() should be the last transformation and it should be applied to both the input and target images
transform_target = None
