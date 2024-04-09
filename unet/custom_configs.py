'''
These configs are optional and can be used to customize the model training process.
They are most useful in adapting the model to different datasets or tasks.
They are not usually required for the model to work, but can be used in special cases.
Leave this file empty unless you know better.
'''

import cv2
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import BasicTransform
import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch import nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def TARGET_READER(path: str, _: int):
    # Load the mask from a .npy file
    x = np.load(path)['mask']

    return x


def INPUT_READER(x, channels):
    img = Image.open(x)
    if img.mode in [
            "I", "I;16", "I;16B", "I;16L"]:  # For 16-bit grayscale images
        if channels == 3:  # If expecting RGB output, convert accordingly
            img = img.convert("RGB")
        else:  # Keep as is for grayscale
            img = img.convert("I")
    elif img.mode not in ["RGB", "L"]:  # If not standard 8-bit modes, convert
        img = img.convert("RGB" if channels == 3 else "L")
    return np.array(img)


# Set the task to segmentation for utility functions which depend on this value
TASK = 'segmentation'


def weights(
        class_1: int, class_2: int, composite: int, bg_freq: float = 0.8,
        class_2_freq: float = 0.6, class_1_freq: float = 0.4, device=DEVICE):
    '''
    Calculate the weights for the classes based on the number of images in each class 
    and the frequency of the classes in the composite images. By default class_1 is weighted
    heavier than class_2 and the background class.

    Parameters:
    ----------
    class_1: int
        Number of images in class 1
    class_2: int
        Number of images in class 2
    composite: int
        Number of composite images
    bg_freq: float
        Frequency of the background class in the images
        Default: 0.8
    class_2_freq: float
        Frequency of class 2 in the composite images
        Default: 0.6
    class_1_freq: float
        Frequency of class 1 in the composite images
        Default: 0.4
    '''
    # Set the weights for the classes
    # Total number of images contributing to the classes (class 2, class 1, background)
    total_images = class_1 + class_2 + composite
    class_2_weight = class_2 + composite * class_2_freq  # Weight of class 2
    class_1_weight = class_1 + composite * class_1_freq  # Weight of class 1
    background_weight = total_images * bg_freq  # Weight of the background

    # Normalize to get frequencies
    total_weight = class_1_weight + class_2_weight + background_weight
    class_2_freq = class_2_weight / total_weight
    class_1_freq = class_1_weight / total_weight
    background_freq = background_weight / total_weight

    # Inverse of the frequencies as weights
    # The lower the frequency, the higher the weight
    weights = [1.0 / freq
               for freq in [background_freq, class_1_freq, class_2_freq]]
    # Normalize the weights to the smallest weight
    normalized_weights = [weight / min(weights) for weight in weights]

    return torch.tensor(normalized_weights, dtype=torch.float).to(device)


# Set the loss function to CrossEntropyLoss for segmentation tasks
def LOSS_FN(x, y):
    y = y.squeeze(1)  # Remove the channel dimension
    return nn.CrossEntropyLoss()(x, y)


CBAR = False


class ToTensorWithDtype(BasicTransform):
    """Convert image, mask, or any additional targets to `torch.Tensor` and allows casting to specified dtype."""

    def __init__(self, always_apply=True, dtype=torch.float):
        super(ToTensorWithDtype, self).__init__(always_apply, p=1)
        self.dtype = dtype

    def __call__(self, force_apply=False, **kwargs):
        for key, item in kwargs.items():
            if item is not None:
                kwargs[key] = torch.tensor(item, dtype=self.dtype)
        return kwargs

    def get_transform_init_args_names(self):
        return ("dtype",)

    def get_params(self):
        return {"dtype": self.dtype}

    def get_params_dependent_on_targets(self, params):
        # Here you can implement logic to dynamically choose dtype based on target types, if necessary.
        return params

    @property
    def targets_as_params(self):
        # Define which targets are used to compute the parameters
        return []

    def get_transform_init_args_names(self):
        return ()


IMAGE_SIZE = 256  # Desired size. Adjust as needed.

transform_both = None
transform_input = A.Compose(
    [
        # Normalize the image but not the mask
        A.Normalize(mean=[0.], std=[1.], max_pixel_value=2 **
                    16 - 1, always_apply=True),
        ToTensorWithDtype(dtype=torch.float64),  # 16-bit images
    ],
)

transform_target = A.Compose(
    [
        ToTensorWithDtype(dtype=torch.long),  # 8-bit masks
    ],
)

# Adjust these parmaters to affect the training data
# Glob pattern for training images
TRAIN_IMG_PATTERN = "/scratch/general/nfs1/u0977428/transfer/preprocess/tifs/*.tif"
# Glob pattern for target images
TARGET_IMG_PATTERN = "/scratch/general/nfs1/u0977428/transfer/preprocess/masks/*.npz"

CHANNELS_INPUT = 1  # Grayscale
CHANNELS_OUTPUT = 3  # 3 channels for the mask

DATASET_TO_FLOAT = False  # Handle type conversion independently in the transforms

SKIP_CHANNEL_EXPANSION = True  # Skip adding a channel dimension if not present
SAVE_DST = True
DST_SAVE_DIR = "/scratch/general/nfs1/u0977428/Training/unet/datasets"  # directory to save the datasets


LOAD_MODEL = True  # set to True if you want to load a pre-trained model
CHECKPOINT = r"unet.pth.tar"
EXAMPLES_DIR = r'results'


def ALIGNMENT_FN(x, y):
    return x == '.'.join(y.split('.')[:-1]) if 'composite' not in x else x.split('.')[0] == y.split('.')[0] 
