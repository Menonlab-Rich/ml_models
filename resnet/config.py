import torch
import albumentations as A
from utils import is_notebook


IMG_CHANNELS = 3
NUM_CLASSES = 2  # binary classification
IMAGE_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOSS_FN = lambda x, y: torch.nn.functional.cross_entropy(x, y.long(), weight=WEIGHTS)
WEIGHTS = torch.tensor([1, 1]) # class weights for the loss function must be the same size as the number of classes
TRAIN_IMG_DIR = "data/train/images"
TRAIN_ANNOTATIONS_FILE = "data/train/annotations.json"
CHECKPOINT = "model.pth"
LOAD_MODEL = False
SAVE_MODEL = True
PREDICT_ONLY = False
EXAMPLES_DIR = "examples"
LOSSES_FILE = "losses.npz"
IN_JUPYTER = is_notebook()
LOSS_PLOT = "loss.png"

TRANSFORMS = A.Compose(
    [A.to_tensor(),],
    bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
