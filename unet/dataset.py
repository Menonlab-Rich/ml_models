from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch import tensor
from glob import glob
import logging


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
        self.images, self.targets = self._load_data(
            input_globbing_pattern, target_globbing_pattern)
        # Store the transformations to be applied to the images
        self.transform = transforms
        # Parse the arguments passed to the constructor
        self._parse_args(kwargs)
        # Log the number of images found
        self.logger.info(f"Found {len(self.images)} images")
        # Log the number of target images found
        self.logger.info(f"Found {len(self.targets)} target images")

    def _parse_args(self, kwargs):
        '''
        Parse the arguments passed to the constructor
        '''
        defaults = {
            "target_input_combined": False, "logger": logging.getLogger(
                __name__),
            "channels": (3, 3),
            "input_reader": lambda x, channels: np.array(Image.open(x).convert(
                "RGB" if channels == 3 else "L")),
            "target_reader": lambda x, channels: np.array(Image.open(x).convert(
                "RGB" if channels == 3 else "L")), 
            "transform_keys": {"input": "image", "target": "image", "both": ("image", "target")}
            }
        # Store the arguments as attributes of the defaults object
        for key, value in kwargs.items():
            defaults[key] = value

        # Store the attributes of the defaults object as attributes of the Dataset object
        for key, value in defaults.items():
            setattr(self, key, value)

    def _load_data(self, input_globbing_pattern, target_globbing_pattern):
        inputs = glob(input_globbing_pattern, recursive=True)
        targets = glob(target_globbing_pattern, recursive=True)
        assert len(inputs) == len(
            targets), "Number of images and targets must be equal"
        return inputs, targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
       # Open the image and convert it to RGB
        input_channels, target_channels = self.channels
        input_ = self.input_reader(self.images[idx], input_channels)
        target_ = self.target_reader(self.targets[idx], target_channels)

        # If transform is specified, apply the transformations to the images
        if self.transform:
            input_only = self.transform_keys["input"]
            target_only = self.transform_keys["target"]
            both = self.transform_keys["both"]
            
            # Apply the transformations to the images one by one
            # This is done to support different transformations for the input and target images
            # By converting the list to an iterator, we can use the next() function to apply the transformations
            # Without having to separately check each transformation
            it = iter(self.transform)
            # Apply the first transformation to the input only
            if next(it):
                augmentations = self.transform[0](**{input_only: input_})
                input_ = augmentations["image"]
            # Apply the second transformation to the target only
            if next(it):
                augmentations = self.transform[1](**{target_only: target_})
                target_ = augmentations["image"]
            
            # Apply the third transformation to both the image and the target
            if next(it):
                augmentations = self.transform[2](**{both[0]: input_, both[1]: target_})
                input_ = augmentations[both[0]]
                target_ = augmentations[both[1]]

        # Transpose the images to channel-first format if necessary

        # Log the shapes of the images
        self.logger.debug("Input shape: {}".format(input_.shape))
        self.logger.debug("Target shape: {}".format(target_.shape))
        
        # Return the images as tensors
        return tensor(input_).float(), tensor(target_).float()
