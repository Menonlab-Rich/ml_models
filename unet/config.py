import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

# Do not change these parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Adjust these parameters to affect the training process. 
LEARNING_RATE = 2e-4 
BATCH_SIZE = 16
NUM_EPOCHS = 6 # number of epochs to train the model
NUM_WORKERS = 8 # based on number of cores
CHANNELS_INPUT = 3 # Grayscale
CHANNELS_OUTPUT = 3 # RGB
LOSS_FN = lambda x, y: nn.L1Loss()(x, y) * 100 # L1 loss with a weight of 100

# Adjust these parmaters to affect the training data
IMAGE_SIZE = 256 # size of your input images, you can increase it if you have large memory
TRAIN_IMG_PATTERN="/scratch/general/nfs1/u0977428/Training/jpg/IN*.jpg" # Glob pattern for training images
TARGET_IMG_PATTERN="/scratch/general/nfs1/u0977428/Training/jpg/OUT*.jpg" # Glob pattern for target images

# Adjust these parameters to affect plotting results
CMAP_IN = 'gray' if CHANNELS_INPUT == 1 else None # colormap for input images
CMAP_OUT = 'inferno' if CHANNELS_OUTPUT == 1 else None # colormap for output images
PLOTTING_INTERPOLATION = lambda ch: 'nearest' if ch == 1 else 'bilinear' # interpolation for plotting images
CBAR_MIN = lambda x: x.min() # Function to calculate the colorbar minimum. Can be set to None for automatic calculation, or a number for manual setting
CBAR_MAX = lambda x: x.max() # Function to calculate the colorbar maximum Can be set to None for automatic calculation, or a number for manual setting


# Adjust these parameters to affect model saving and loading
EXAMPLES_DIR="/scratch/general/nfs1/u0977428/Training/ml_models/unet/results" # Where to save example images
LOAD_MODEL = False # set to True if you want to load a pre-trained model
SAVE_MODEL = True # set to True to save the model
CHECKPOINT = "unet.pth.tar" # Saved modle filename



# Augmentation pipeline
# Find documentation here: https://albumentations.ai/docs/
# This pipeline is applied to both the input and target images
transform_both = A.Compose(
    [
        # A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
         A.Normalize(
             mean=[0., 0., 0.], std=[1., 1., 1.], max_pixel_value=255.0,
         ),
         A.LongestMaxSize(max_size=IMAGE_SIZE, always_apply=True), # Scale the image to the maximum size
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
transform_target = [] # No transformations for the target images by default


