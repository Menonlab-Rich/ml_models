'''
These configs are optional and can be used to customize the model training process.
They are most useful in adapting the model to different datasets or tasks.
They are not usually required for the model to work, but can be used in special cases.
Leave this file empty unless you know better.
'''

import numpy as np
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def TARGET_READER(path: str, _: int):
    # Load the mask from a .npy file
    x = np.load(path)['mask']

    # Map class labels: -1 -> 1 (class of interest 1), 0 -> 0 (background), 1 -> 2 (class of interest 2)
    target_mapped = np.where(x == -1, 1, np.where(x == 0, 0, 2))

    return target_mapped


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
    return nn.CrossEntropyLoss(weight=weights(3422, 1744, 1000))(x, y)

CBAR = False

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

IMAGE_SIZE = 256  # Desired size. Adjust as needed.

transform_both = None
tranform_input = A.Compose(
    [
        # Resize images to a fixed size. Use interpolation='nearest' for masks in additional_targets.
        A.LongestMaxSize(max_size=IMAGE_SIZE, always_apply=True, interpolation=cv2.INTER_LINEAR),
        # Normalize the image but not the mask
        A.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.], max_pixel_value=255.0),
        ToTensorV2(),
    ],
)

transform_target = A.Compose(
    [
        # Resize images to a fixed size. Use interpolation='nearest' for masks in additional_targets.
        A.LongestMaxSize(max_size=IMAGE_SIZE, always_apply=True, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(),
    ],
)
