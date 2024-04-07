import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import importlib.util
import os

'''
UNet configurations
These configurations represent the default settings for the UNet model.
They can be customized by creating a custom_configs.py file in the same directory
If the custom_configs.py file is present, the configurations in it will override the default configurations
Do NOT change the default configurations here. 
Instead, create a custom_configs.py file and change the configurations there.
'''

# Set the device to cuda if available, otherwise use cpu (probably good not to change this)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Function to read the target images. This function is used to read the target images during training if you want to customize the reading process
TARGET_READER = None
# Function to read the input images. This function is used to read the input images during training if you want to customize the reading process
INPUT_READER = None
# Task to perform. Options are 'classification', 'segmentation', 'translation
TASK = 'translation'
# Whether to convert the dataset to float. The default behavior is to convert the dataset to float. 
# If you disable this, you will have to convert the dataset to the appropriate data type elsewhere in the pipeline
# We suggest a custom INPUT_READER or TARGET_READER function for this purpose or custom transformations
# Generally, it is best to leave this as True
DATASET_TO_FLOAT=True

# Adjust these parameters to affect the training process.
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 6  # number of epochs to train the model
NUM_WORKERS = 8  # based on number of cores
CHANNELS_INPUT = 3  # RGB
CHANNELS_OUTPUT = 3  # RGB

# The loss function tells the model how well it is doing. You can customize this function to suit your task
def LOSS_FN(x, y): return nn.L1Loss()(
    x, y) * 100  # L1 loss with a weight of 100


# Adjust these parmaters to affect the training data
IMAGE_SIZE = 256  # size of your input images, you can increase it if you have large memory
# Glob pattern for training images
TRAIN_IMG_PATTERN = "/scratch/general/nfs1/u0977428/Training/jpg/IN*.jpg"
# Glob pattern for target images
TARGET_IMG_PATTERN = "/scratch/general/nfs1/u0977428/Training/jpg/OUT*.jpg"

# Adjust these parameters to affect plotting results
CMAP_IN = 'gray' if CHANNELS_INPUT == 1 else None  # colormap for input images
CMAP_OUT = 'inferno' if CHANNELS_OUTPUT == 1 else None  # colormap for output images
# interpolation for plotting images
def PLOTTING_INTERPOLATION(ch): return 'nearest' if ch == 1 else 'bilinear'
CBAR = True  # Whether to show the colorbar
# Function to calculate the colorbar minimum. Can be set to None for automatic calculation, or a number for manual setting
def CBAR_MIN(x): return x.min()
# Function to calculate the colorbar maximum Can be set to None for automatic calculation, or a number for manual setting
def CBAR_MAX(x): return x.max()


# Adjust these parameters to affect model saving and loading
# Where to save example images
EXAMPLES_DIR = "/scratch/general/nfs1/u0977428/Training/ml_models/unet/results"
LOAD_MODEL = False  # set to True if you want to load a pre-trained model
SAVE_MODEL = True  # set to True to save the model
CHECKPOINT = "unet.pth.tar"  # Saved modle filename


# Augmentation pipeline
# Find documentation here: https://albumentations.ai/docs/
# This pipeline is applied to both the input and target images
transform_both = A.Compose(
    [
        # A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.Normalize(
            mean=[0., 0., 0.], std=[1., 1., 1.], max_pixel_value=255.0,
        ),
        # Scale the image to the maximum size
        A.LongestMaxSize(max_size=IMAGE_SIZE, always_apply=True),
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

# This pipeline is applied to the input images only
# You can add additional transformations to the input images if you want
# Just make sure not to add ToTensorV2() to the input transformations
# This is because ToTensorV2() should be the last transformation and it should be applied to both the input and target images
transform_input = A.Compose(
    [
        # add noise to the input image with a probability of 50%
        A.GaussNoise(p=0.5),
    ]
)

# This pipeline is applied to the target images only
# You can add transformations to the target images if you want by following the same pattern.
# Just make sure not to add ToTensorV2() to the input transformations
# This is because ToTensorV2() should be the last transformation and it should be applied to both the input and target images
transform_target = []  # No transformations for the target images by default

# Path to the custom_configs.py file
custom_config_path = (lambda: os.path.join(os.path.dirname(__file__), "custom_configs.py"))()
# Customize the configurations by importing the custom_configs.py file
if os.path.isfile(custom_config_path):
    spec = importlib.util.spec_from_file_location("custom_config", custom_config_path)
    custom_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_config)
    
    # Update the globals() dictionary with the settings from custom_config
    # This effectively overwrites any existing settings with those from custom_config
    globals().update({k: v for k, v in vars(custom_config).items() if not k.startswith("__")})
