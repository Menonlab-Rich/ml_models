from torch import nn
import os
import torch
from glob import glob
from skimage import io
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from dataset import GenericLoader


class ToTensorWithDtype(A.BasicTransform):
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


_root_dir = r'D:\CZI_scope\code\ml_models\cnn'
_data_dir = r'D:\CZI_scope\code\preprocess_training'

input_pattern = os.path.join(_data_dir, 'tifs', '605-*.tif')
target_pattern = os.path.join(_data_dir, 'masks', '605-*.npz')
num_epochs = 6
label_smoothing = 0.1  # Label smoothing factor
loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
lr = 2e-4  # Learning rate
batch_size = 32  # Batch size
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class InputLoader(GenericLoader):
    def __init__(self, pattern):
        self.pattern = pattern
        self.files = sorted(glob(pattern))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self._read(self.files[idx])
    
    def __iter__(self):
        for file in self.files:
            yield self._read(file)
    def get_ids(self, i=None):
        if i is not None:
            return self.files[i]
        return self.files
    
    def _read(self, file):
        '''
        Read a 16-bit tif file as black and white
        '''
        return io.imread(file, as_gray=True)
    
class TargetLoader(GenericLoader):
    def __init__(self, pattern):
        self.pattern = pattern
        self.files = sorted(glob(pattern))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return self._read(self.files[idx])
    
    def __iter__(self):
        for file in self.files:
            yield self._read(file)
    def get_ids(self, i=None):
        if i is not None:
            return self.files[i]
        return self.files
    
    def _read(self, file):
        '''
        Read a numpy file containing a mask
        '''
        return np.load(file)['mask']
        
    


input_loader = InputLoader(input_pattern)
target_loader = TargetLoader(target_pattern)


input_transform = A.Compose([
    # Precalculated statistics for normalization
    A.Normalize(
        mean=[641.6],
        std=[3606.7],
        max_pixel_value=2 ** 16 - 1, always_apply=True),
    A.ShiftScaleRotate(shift_limit=0.0625,
                       scale_limit=0.1, rotate_limit=0, p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    ToTensorWithDtype(dtype=torch.float64),
])

target_transform = A.Compose([
    ToTensorV2(),
])

transforms = {
    'input': lambda x: input_transform(image=x)['image'],
    'target': lambda x: target_transform(mask=x)['mask'],
}
width = 820
height = 460
nclasses = 3


def scheduler(optimizer): return torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=2, gamma=0.1)


logger = logging.getLogger(__name__)
dset_dir = os.path.join(_root_dir, 'datasets')
results_dir = os.path.join(_root_dir, 'results')
checkpoint_dir = os.path.join(_root_dir, 'checkpoints')

if __name__ == 'config':
    # Create directories for datasets, results, and checkpoints
    _logger = logging.getLogger(__name__ + '.config')
    _logger.setLevel(logging.INFO)

    os.makedirs(dset_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    _logger.info('Created directories for datasets, results, and checkpoints.')
    # delete the logger
    del _logger
