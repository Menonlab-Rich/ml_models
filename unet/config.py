import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 8 # based on number of cores
IMAGE_SIZE = 256
CHANNELS_INPUT = 3
CHANNELS_OUTPUT = 3
L1_LAMBDA = 100
NUM_EPOCHS = 6
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT = "unet.pth.tar"
TRAIN_IMG_DIR="data/landscapes/gray"
TARGET_DIR="data/landscapes/color"
EXAMPLES_DIR="results"


training_transform = A.Compose(
    [
        A.Normalize(
            mean=[0., 0., 0.], std=[1., 1., 1.], max_pixel_value=255.0,
        ),
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        ToTensorV2(),
    ],
    additional_targets={'target': 'image'} # required if target is an image
)


