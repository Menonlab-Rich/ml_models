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
CHANNELS_INPUT = 3
CHANNELS_OUTPUT = 3
STDEV = 0.5
MEAN = 0.5

from albumentations.core.transforms_interface import ImageOnlyTransform


class ToGrayscale1Channel(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(ToGrayscale1Channel, self).__init__(always_apply, p)

    def apply(self, image, **params):
        # Convert the image to grayscale with OpenCV
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Optionally, add a new axis to make the shape (H, W, 1) instead of (H, W)
        # gray_image = gray_image[:, :, None]
        return gray_image


both_transform = A.Compose(
    [
        ToGrayscale1Channel(p=1), # convert to grayscale with a single channel
        A.Normalize(mean=[MEAN], std=[STDEV], max_pixel_value=255.)   
    ], additional_targets={"target": "image"},
)

transform_only_input = A.Compose(
    [
        ToTensorV2(),
    ]
)

transform_only_target = A.Compose(
    [
        ToTensorV2(),
    ]
)
