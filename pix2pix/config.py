import torch
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 8
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 9
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
CHANNELS_INPUT = 1
CHANNELS_OUTPUT = 1
STDEV = 0.5
MEAN = 0.5

from albumentations.core.transforms_interface import ImageOnlyTransform



def threshold(image, thresh=47, **_):
    image[image <= thresh] = 0
    return image

both_transform = A.Compose(
    [
    ], additional_targets={"target": "image"},
)

transform_only_input = A.Compose(
    [
        A.Normalize(mean=[MEAN], std=[STDEV], max_pixel_value=1.),
        ToTensorV2(),
    ]
)

transform_only_target = A.Compose(
    [
        A.Normalize(mean=[MEAN], std=[STDEV], max_pixel_value=1.),
        ToTensorV2(),
    ]
)
