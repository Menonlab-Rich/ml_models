from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch import tensor
from glob import glob
import logging
import warnings
import config

def default_reader(x, channels):
    return np.array(Image.open(x).convert("RGB" if channels == 3 else "L"))


class Dataset(Dataset):
    def __init__(self, input_globbing_pattern=None,
                 target_globbing_pattern=None,
                 transforms=(None, None, None), **kwargs):
        '''
        Initializes the Dataset object that can be used with PyTorch's DataLoader

        Parameters:
        ----------
        input_globbing_pattern: str
            Globbing pattern to find the input data
        target_globbing_pattern: str
            Globbing pattern to find the target data
        transforms: [torchvision.transforms]
            Transformations to be applied to the images. In the order of (input, target, both)
            Default: (None, None, None)
        logger: logging.Logger
            Logger to be used for logging
            Default: logging.getLogger(__name__)
        channels: tuple (int, int)
            Number of channels in the input and target images respectively
            Default: (3, 3)
        input_reader: function(filename: str, channels: int) -> np.ndarray | torch.Tensor
            Function to read the input data
            default: lambda x, channels: Image.open(x).convert("RGB" if channels == 3 else "L")
        target_reader: function(str, int) -> np.ndarray | torch.Tensor
            Function to read the target data
            default: lambda x, channels: Image.open(x).convert("RGB" if channels == 3 else "L")

        '''
        # Store the paths to the images
        self._defaults = None
        self._filenames = [] 
        validate_alignment = False
        alginment_fn = None
        if getattr(config, 'VALIDATE_ALIGNMENT', False):
            validate_alignment = True
        if getattr(config, 'ALIGNMENT_FN', None):
            validate_alignment = True
            alignment_fn = config.ALIGNMENT_FN
        self.images, self.targets = self._load_data(
            input_globbing_pattern, target_globbing_pattern, validate_alignment,
            alignment_fn)
        # Store the transformations to be applied to the images
        self.transform = transforms
        # Parse the arguments passed to the constructor
        self._parse_args(kwargs)
        # Log the number of images found
        self.logger.info(f"Found {len(self.images)} images")
        # Log the number of target images found
        self.logger.info(f"Found {len(self.targets)} target images")

    def set_args(self, **kwargs):
        '''
        Set the argument for the Dataset object
        '''
        self._parse_args(kwargs)

    def _parse_args(self, kwargs):
        '''
        Parse the arguments passed to the constructor
        '''
        if self._defaults is not None:
            # Update the defaults with the new arguments
            self._defaults.update(kwargs)
            kwargs = self._defaults

        defaults = {
            "target_input_combined": False, "logger": logging.getLogger(
                __name__),
            "channels": (3, 3),
            "input_reader": default_reader,
            "target_reader": default_reader,
            "transform_keys": {"input": "image", "target": "image", "both": ("image", "target")},
            "to_float": True,  # Convert the images to float by default
            "validate_alignment": False,
            "alignment_fn": None
        }
        # Store the arguments as attributes of the defaults object
        for key, value in kwargs.items():
            defaults[key] = value if key in defaults else defaults[key]

        # Store the attributes of the defaults object as attributes of the Dataset object
        for key, value in defaults.items():
            setattr(self, key, value)

        self._defaults = defaults  # save the defaults so we can update them later if needed

    def _load_data(
            self, input_globbing_pattern, target_globbing_pattern,
            validate_alignment=False, alignment_fn=None):
        inputs = sorted(glob(input_globbing_pattern, recursive=True))
        targets = sorted(glob(target_globbing_pattern, recursive=True))
        assert len(inputs) == len(
            targets), "Number of images and targets must be equal"
        if validate_alignment:
            from os import path
            if not alignment_fn:
                alignment_fn = lambda x, y: x == y # default behavior is to check if the filenames are the same
            for i, t in zip(inputs, targets):
                name_i = path.basename(i)
                name_t = path.basename(t)
                assert alignment_fn(name_i, name_t), f"Input and target images must be aligned. Found {name_i} and {name_t}"
        return inputs, targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        input_channels, target_channels = self.channels
        input_ = self.input_reader(self.images[idx], input_channels)
        target_ = self.target_reader(self.targets[idx], target_channels)
        if self.transform:
            # Check if the transform is a list (legacy behavior)
            if isinstance(
                    self.transform, list) or isinstance(
                    self.transform, tuple):
                warnings.warn(
                    "Using a list for transformations is deprecated. Use a dictionary with keys 'input', 'target', and 'both' for clearer and more robust transformation application.",
                    DeprecationWarning)

                it = iter(self.transform)
                try:
                    # Apply the first transformation to the input only
                    if next(it):
                        augmentations = self.transform[0](image=input_)
                        input_ = augmentations['image']
                    # Apply the second transformation to the target only
                    if next(it):
                        augmentations = self.transform[1](image=target_)
                        target_ = augmentations['image']
                    # Apply the third transformation to both the input and the target
                    if next(it):
                        augmentations = self.transform[2](
                            image=input_, target=target_)
                        input_ = augmentations['image']
                        target_ = augmentations['target']
                except StopIteration:
                    pass
            # New recommended behavior using a dictionary
            elif isinstance(self.transform, dict):
                if 'input' in self.transform:
                    augmented = self.transform['input'](image=input_)
                    input_ = augmented['image']
                if 'target' in self.transform:
                    augmented = self.transform['target'](image=target_)
                    target_ = augmented['image']
                if 'both' in self.transform:
                    augmented = self.transform['both'](
                        image=input_, target=target_)
                    input_ = augmented['image']
                    target_ = augmented['target']
            else:
                raise TypeError(
                    "Transform must be either a list (deprecated) or a dictionary.")

        # Log the shapes of the images
        self.logger.debug("Input shape: {}".format(input_.shape))
        self.logger.debug("Target shape: {}".format(target_.shape))

        # Ensure input and target are tensors (considering they might already be tensors after transformation)
        input_tensor, target_tensor = map(tensor, (input_, target_))
        if self.to_float:
            input_tensor = input_tensor.float()
            target_tensor = target_tensor.float()
        # Store a mapping from the tensor to the filename
        # For plotting purposes
        self._filenames.append((self.images[idx], self.targets[idx]))
        return input_tensor, target_tensor, 
    
    def get_filenames(self, i=None):
        return self._filenames if i is None else self._filenames[i]
