from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import config
from torch import tensor
import math
import logging
from glob import glob
# class Dataset(Dataset):
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.files = os.listdir(root_dir)
#         self.files = [os.path.join(root_dir, file) for file in self.files]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         # adding basename ensures that the fn works for relative and absolute paths
#         img_path = os.path.join(self.root_dir, os.path.basename(self.files[idx]))
#         img = Image.open(img_path)
#         img = np.array(img)
#         input_img = img[:, :256, :]
#         target_img = img[:, 256:, :]
#         augmentations = config.both_transform(image=input_img, image0=target_img)
#         input_img, target_img = augmentations["image"], augmentations["image0"]

#         input_img = config.transform_only_input(image=input_img)["image"]
#         target_img = config.transform_only_target(image=target_img)["image"]

#         return input_img, target_img


class Dataset(Dataset):
    def __init__(self, image_globbing_pattern=None,
                 target_globbing_pattern=None,
                 transform=(None, None, None), **kwargs):
        '''
        Initializes the Dataset object that can be used with PyTorch's DataLoader

        Parameters:
        ----------
        image_globbing_pattern: str
            Globbing pattern to find the input images
        target_globbing_pattern: str
            Globbing pattern to find the target images
        transform: [torchvision.transforms]
            Transformations to be applied to the images. In the order of (both, input, target)
            Default: (None, None, None)
        target_input_combined: bool
            If True, the input and target images are assumed to be combined into a single image. 
            The input image is placed on the left and the target image is placed on the right.
            Default: False
        logger: logging.Logger
            Logger to be used for logging
            Default: logging.getLogger(__name__)
        '''
        # Store the paths to the images
        self.images, self.targets = self._load_images(
            image_globbing_pattern, target_globbing_pattern)
        # Store the transformations to be applied to the images
        self.transform = transform
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
            "target_input_combined": False,
            "logger": logging.getLogger(__name__),
        }
        # Store the arguments as attributes of the defaults object
        for key, value in kwargs.items():
            defaults[key] = value
        
        # Store the attributes of the defaults object as attributes of the Dataset object
        for key, value in defaults.items():
            setattr(self, key, value)
    
    def _load_images(self, image_globbing_pattern, target_globbing_pattern):
        images = glob(image_globbing_pattern, recursive=True)
        targets = glob(target_globbing_pattern, recursive=True)
        assert len(images) == len(
            targets), "Number of images and targets must be equal"
        return images, targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
       # Open the image and convert it to RGB
        img = Image.open(self.images[idx]).convert("RGB")

        # If target_input_combined is True, split the image into two halves
        if self.target_input_combined:
            img_width = img.size[0]
            target_img = img[:, img_width//2:, :]
            img = img[:, :img_width//2, :]
        else:
            target_img = Image.open(self.targets[idx]).convert("RGB")

        # Convert the images to numpy arrays and normalize them
        img = (np.array(img).astype(np.uint8) / 255).astype(np.float32)
        target_img = (np.array(target_img).astype(
            np.uint8) / 255).astype(np.float32)

        # If transform is specified, apply the transformations to the images
        if self.transform:
            it = iter(self.transform)
            # Apply the first transformation to both the image and the target
            if next(it):
                augmentations = self.transform[0](image=img, target=target_img)
                img = augmentations["image"]
                target_img = augmentations["target"]
            # Apply the second transformation to the image only
            if next(it):
                augmentations = self.transform[1](image=img)
                img = augmentations["image"]
            # Apply the third transformation to the target only
            if next(it):
                augmentations = self.transform[2](image=target_img)
                target_img = augmentations["image"]

        # Transpose the images to channel-first format if necessary
        if img.shape[2] == config.CHANNELS_INPUT:
            img = np.transpose(img, (2, 0, 1))
        if target_img.shape[2] == config.CHANNELS_OUTPUT:
            target_img = np.transpose(target_img, (2, 0, 1))

        # Log the shapes of the images
        self.logger.debug("Image shape: {}".format(img.shape))
        self.logger.debug("Target shape: {}".format(target_img.shape))

        # Return the images as tensors
        return tensor(img), tensor(target_img)