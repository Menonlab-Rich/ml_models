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
        make_even: bool
            Adjust the dimensions of the images to make them even
            Default: False
        make_square: bool
            Adjust the dimensions of the images to make them square
            Default: False
        pad_or_crop: str
            If "pad", pad the images to make them even or square. If "crop", crop the images to make them even or square.
            Default: "pad"
        target_input_combined: bool
            If True, the input and target images are assumed to be combined into a single image. 
            The input image is placed on the left and the target image is placed on the right.
            Default: False
        logger: logging.Logger
            Logger to be used for logging
            Default: logging.getLogger(__name__)
        match_shape: bool
            If True, the shapes of the input and target images are matched before applying any other transformations.
            They are matched by the method specified by the pad_or_crop parameter.
            Default: False
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
            "make_even": False,
            "make_square": False,
            "pad_or_crop": "pad",
            "target_input_combined": False,
            "logger": logging.getLogger(__name__),
            "match_shape": False,
            "axis": "y" # Default axis for splitting the image
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
        img = (np.array(img).astype(np.uint8))

        # If target_input_combined is True, split the image into two halves
        if self.target_input_combined:
            if self.axis == "y":
                img, target_img = self._split_image_y(img)
            else:
                img, target_img = self._split_image_x(img)
        else:
            target_img = Image.open(self.targets[idx]).convert("RGB")

        # Convert the images to numpy arrays and normalize them
        img = (np.array(img).astype(np.uint8) / 255).astype(np.float32)
        target_img = (np.array(target_img).astype(
            np.uint8) / 255).astype(np.float32)

        if self.match_shape:
            img, target_img = self._match_shape(img, target_img)
        
        # If make_even is True, adjust the dimensions of the images to make them even
        if self.make_even:
            img = self._make_even(img)
            target_img = self._make_even(target_img)

        # If make_square is True, adjust the dimensions of the images to make them square
        if self.make_square:
            img = self._make_square(img)
            target_img = self._make_square(target_img)

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

    def _split_image_y(self, img: np.ndarray):
        '''
        Split the image into two halves along the y-axis
        '''
        img_width = img.shape[1]
        target_img = np.copy(img[:, img_width//2:, :])
        img = np.copy(img[:, :img_width//2, :])
        return img, target_img
    
    def _split_image_x(self, img: np.ndarray):
        '''
        Split the image into two halves along the x-axis
        '''
        img_height = img.shape[0]
        img = np.copy(img[img_height//2:, :, :])
        target_img = np.copy(img[:img_height//2, :, :])
        return img, target_img

    def _match_shape(self, img: np.ndarray, target_img: np.ndarray):
        if self.pad_or_crop not in ["pad", "crop"]:
            raise ValueError("pad_or_crop must be one of 'pad' or 'crop'")
        
        if self.pad_or_crop == "pad":
            return self._match_shape_with_padding(img, target_img)
        
        return self._match_shape_with_cropping(img, target_img)

    def _match_shape_with_cropping(self, img: np.ndarray, target_img: np.ndarray):
        # compare the shapes of the images
        img_height, img_width = img.shape[:2]
        target_height, target_width = target_img.shape[:2]
        
        if img_height == target_height and img_width == target_width:
            return img, target_img
        
        if img_height < target_height:
            # crop the target image
            target_img = target_img[:img_height, :, :]
        else:
            # crop the input image
            img = img[:target_height, :, :]
        
        if img_width < target_width:
            # crop the target image
            target_img = target_img[:, :img_width, :]
        else:
            # crop the input image
            img = img[:, :target_width, :]
        
        return img, target_img
    
    def _match_shape_with_padding(self, img: np.ndarray, target_img: np.ndarray):
        # compare the shapes of the images
        img_height, img_width = img.shape[:2]
        target_height, target_width = target_img.shape[:2]
        
        if img_height == target_height and img_width == target_width:
            return img, target_img

        if img_height > target_height:
            # pad the target image
            target_img = self._pad_to(target_img, 0, img_height)
        
        else:
            # pad the input image
            img = self._pad_to(img, 0, target_height)
        
        if img_width > target_width:
            # pad the target image
            target_img = self._pad_to(target_img, 1, img_width)
        else:
            # pad the input image
            img = self._pad_to(img, 1, target_width)
        
        return img, target_img
        
        
    def _pad_to(self, x: np.ndarray, dim, size: int):
        '''
        Given an image, a dimension, and a target size, pad the image to make the dimension equal to the target size.
        '''
        
        # if dimension is zero, pad the image along the height dimension
        if dim == 0:
            return np.pad(x, [(0, size - x.shape[0]), (0, 0), (0, 0)], mode='constant')
        
        if dim == 1:
            return np.pad(x, [(0, 0), (0, size - x.shape[1]), (0, 0)], mode='constant')
        
        if dim == 2:
            return np.pad(x, [(0, 0), (0, 0), (0, size - x.shape[2])], mode='constant')
        
        raise ValueError("dim must be one of 0, 1, or 2")
        

    def _make_even(self, x: np.ndarray):
        '''
        Adjust the dimensions of x to make them even.
        Depending on the value of self.pad_or_crop, this function will either pad or crop the image.

        Parameters:
        ----------
        x: np.ndarray
            The image to be adjusted

        Returns:
        -------
        np.ndarray
            The image with dimensions adjusted to make them even
        '''
        # Get the height and width of the image
        height, width = x.shape[:2]

        # Calculate the difference between the dimension and the next even number
        # If the dimension is already even, the difference is 0
        height_diff = 0 if height % 2 == 0 else 1
        width_diff = 0 if width % 2 == 0 else 1

        # If either dimension is not even, adjust the image
        if height_diff or width_diff:
            # If self.pad_or_crop is set to "pad", pad the image to make the dimensions even
            if self.pad_or_crop == "pad":
                # Create a padding configuration for np.pad. The configuration specifies how many rows/columns to add to each dimension.
                # For the first two dimensions (height and width), it adds 'height_diff' rows and 'width_diff' columns.
                # For the remaining dimensions (if any), it adds no padding (hence the '(0, 0)').
                padding = [(0, height_diff), (0, width_diff)
                           ] + [(0, 0)] * (x.ndim - 2)
                # Pad the array to make the dimensions even. The 'mode' parameter specifies that the padding should be filled with constant values (default is 0).
                x = np.pad(x, padding, mode='constant')
            # If self.pad_or_crop is set to "crop", crop the image to make the dimensions even
            elif self.pad_or_crop == "crop":
                # Crop the array to make the dimensions even. The slicing operation 'x[:height-height_diff, :width-width_diff]' removes 'height_diff' rows from the bottom and 'width_diff' columns from the right.
                x = x[:height-height_diff, :width-width_diff]
            # If self.pad_or_crop is neither "pad" nor "crop", raise a ValueError
            else:
                raise ValueError("pad_or_crop must be one of 'pad' or 'crop'")

        # Return the adjusted image
        return x

    def _make_square(self, x: np.ndarray):
        '''
        Adjust the dimensions of x to make it a perfect square.
        Depending on the value of self.pad_or_crop, this function will either pad or crop the image.

        Parameters:
        ----------
        x: np.ndarray
            The image to be adjusted

        Returns:
        -------
        np.ndarray
            The image with dimensions adjusted to make it a perfect square
        '''
        # If self.pad_or_crop is set to "pad", pad the image to make it a square
        if self.pad_or_crop == "pad":
            return self._pad_to_square(x)
        # If self.pad_or_crop is set to "crop", crop the image to make it a square
        elif self.pad_or_crop == "crop":
            return self._crop_to_square(x)
        # If self.pad_or_crop is neither "pad" nor "crop", raise a ValueError
        else:
            raise ValueError("pad_or_crop must be one of 'pad' or 'crop'")

    def _pad_to_square(self, x: np.ndarray):
        '''
        Pad the image to make it a perfect square (power of 2)
        '''
        # Get the height and width of the image
        height, width = x.shape[:2]

        # Find the larger dimension between height and width
        max_dim = max(height, width)

        # Find the next highest power of 2 from the maximum dimension.
        # This is done because many image processing algorithms perform better or require the image dimensions to be a power of 2.
        padded_dim = 2**math.ceil(math.log2(max_dim))

        # Calculate how much to pad to height and width to make the dimensions equal to padded_dim
        # If the dimension is already equal to padded_dim, no padding is needed for that dimension (hence the 'else 0')
        pad_height = padded_dim - height if height != padded_dim else 0
        pad_width = padded_dim - width if width != padded_dim else 0

        # Create a padding configuration for np.pad. The configuration specifies how many rows/columns to add to each dimension.
        # For the first two dimensions (height and width), it adds 'pad_height' rows and 'pad_width' columns.
        # For the remaining dimensions (if any), it adds no padding (hence the '(0, 0)').
        padding = [(0, pad_height), (0, pad_width)] + [(0, 0)] * (x.ndim - 2)

        # Pad the array to make it square. The 'mode' parameter specifies that the padding should be filled with constant values (default is 0).
        x = np.pad(x, padding, mode='constant')

        # Return the padded image
        return x

    def _crop_to_square(self, x: np.ndarray):
        '''
        Crop the image to make it a perfect square (power of 2)
        '''
        # Get the height and width of the image
        height, width = x.shape[:2]

        # Find the smaller dimension between height and width
        min_dim = min(height, width)

        # Find the next lowest power of 2 from the minimum dimension.
        # This is done because many image processing algorithms perform better or require the image dimensions to be a power of 2.
        cropped_dim = 2**math.floor(math.log2(min_dim))

        # Calculate how much to crop from height and width to make the dimensions equal to cropped_dim
        # If the dimension is already equal to cropped_dim, no cropping is needed for that dimension (hence the 'else 0')
        crop_height = height - cropped_dim if height != cropped_dim else 0
        crop_width = width - cropped_dim if width != cropped_dim else 0

        # Crop the array to make it square. The slicing operation 'x[crop_height:, crop_width:]'
        # removes 'crop_height' rows from the top and 'crop_width' columns from the left.
        x = x[crop_height:, crop_width:]

        # Return the cropped image
        return x
