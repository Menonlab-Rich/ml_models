from PIL import Image
import numpy as np
import os
import config
from torch.utils.data import Dataset
from torch import tensor
import utils
import math



class Dataset(Dataset):
    def __init__(self, image_dir, target_dir,
                 target_naming_pattern=r"{image_fullname}", transform=None,
                 logger=None, make_even=True, make_square=True, pad_or_crop="pad"):
        '''
        Creates a Dataset object that can be used with PyTorch's DataLoader

        Parameters:
        ----------
        image_dir: str
            Path to the directory containing the images
        target_dir: str
            Path to the directory containing the target images
        target_naming_pattern: str, Default: r"{image_fullname}"
            Naming pattern for the target images. The following variables can be used:
            - image_fullname: The full name of the image file including the extension
            - image_name: name of the image file without extension
            - image_ext: extension of the image file
            - image_path: path to the image file

            The variables should be enclosed in curly braces. For example, if the image name is
            "image1.png", then the target image name can be "image1_mask.gif" if the naming
            pattern is "{image_name}_mask.gif"

        transform: torchvision.transforms
            Transformations to be applied to the images. Default: None
        '''
        self.logger = utils.LoggerOrDefault.logger(logger)
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.target_naming_pattern = target_naming_pattern
        self.target_naming_pattern_vars = {
            "image_name": lambda image_name: image_name.split(".")[0],
            "image_ext": lambda image_name: image_name.split(".")[1],
            "image_fullname": lambda image_name: image_name,
            "image_path": lambda image_name: os.path.join(image_dir, image_name)
        }
        self.make_even = make_even
        self.make_square = make_square
        self.pad_or_crop = pad_or_crop
        self.logger.info(f"Found {len(self.images)} images")
        self.logger.info(f"Found {len(os.listdir(target_dir))} target images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # adding basename ensures that the fn works for relative and absolute paths
        img_path = os.path.join(
            self.image_dir, os.path.basename(self.images[idx]))
        target_path = os.path.join(
            self.target_dir, self.target_naming_pattern.format(
                **
                {k: v(os.path.basename(img_path)) for k,
                 v in self.target_naming_pattern_vars.items()}))
        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.float32)

        target_img = Image.open(target_path).convert("RGB")
        target_img = np.array(target_img).astype(np.float32)
        
        # Normalize the images
        # Albumentations setup the normalization for us but we need to do it manually here
        # because we are potentially padding the images before passing them to the model
        # which leads to the padding values being != 0
        # which can confuse the model
        
        img = img / np.max(img)
        target_img = target_img / np.max(target_img)
        
        if self.make_even:
            img = self.make_even(img)
            target_img = self.make_even(target_img)
        if self.make_square:
            img = self.make_square(img)
            target_img = self.make_square(target_img)

        if self.transform:
            augmentations = self.transform(image=img, target=target_img)
            img = augmentations["image"]
            target_img = augmentations["target"]

        # Transpose the images to channel-first format if necessary
        if img.shape[2] == config.CHANNELS_INPUT:
            img = np.transpose(img, (2, 0, 1))
        if target_img.shape[2] == config.CHANNELS_OUTPUT:
            target_img = np.transpose(target_img, (2, 0, 1))

        
        self.logger.debug("Image shape: {}".format(img.shape))
        self.logger.debug("Target shape: {}".format(target_img.shape))
        
        return tensor(img), tensor(target_img)

    def make_even(self, x: np.ndarray):
        '''
        Pad or crop uneven dimensions to make them even
        '''
        height, width = x.shape[:2]
        height_diff = 0 if height % 2 == 0 else 1
        width_diff = 0 if width % 2 == 0 else 1


        if height_diff or width_diff:
            if self.pad_or_crop == "pad":
                # Pad the bottom and right of the image
                padding = [(0, height_diff), (0, width_diff)] + [(0, 0)] * (x.ndim - 2)
                x = np.pad(x, padding, mode='constant')
            elif self.pad_or_crop == "crop":
                # Crop the bottom and right of the image
                x = x[:height-height_diff, :width-width_diff]
            else:
                raise ValueError("pad_or_crop must be one of 'pad' or 'crop'")
        
        return x
        

    def make_square(self, x: np.ndarray):
        '''
        Make the dimensions of x a perfect square
        '''
        if self.pad_or_crop == "pad":
            return self.pad_to_square(x)
        elif self.pad_or_crop == "crop":
            return self.crop_to_square(x)
        else:
            raise ValueError("pad_or_crop must be one of 'pad' or 'crop'")
        
    def pad_to_square(self, x: np.ndarray):
        '''
        Pad the image to make it a perfect square (power of 2)
        '''
        height, width = x.shape[:2]
        max_dim = max(height, width)
        # Find the next power of 2
        padded_dim = 2**math.ceil(math.log2(max_dim))
        pad_height = padded_dim - height if height != padded_dim else 0
        pad_width = padded_dim - width if width != padded_dim else 0
        
        # Pad the array to make it square
        padding = [(0, pad_height), (0, pad_width)] + [(0, 0)] * (x.ndim - 2)
        x = np.pad(x, padding, mode='constant')

        return x
    
    def crop_to_square(self, x: np.ndarray):
        '''
        Crop the image to make it a perfect square (power of 2)
        '''
        height, width = x.shape[:2]
        min_dim = min(height, width)
        # Find the next lowest power of 2
        cropped_dim = 2**math.floor(math.log2(min_dim))
        crop_height = height - cropped_dim if height != cropped_dim else 0
        crop_width = width - cropped_dim if width != cropped_dim else 0
        
        # Crop the array to make it square
        x = x[crop_height:, crop_width:]

        return x
