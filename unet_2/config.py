from base.config import BaseConfigHandler
from base.dataset import Transformer
from os import path as os_path
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Type, List, Union
import torch
import numpy as np
from sklearn.decomposition import PCA

CONFIG_FILE_PATH = 'config.yml'

# path custom tag handler


def path(loader, node):
    seq = loader.construct_sequence(node)
    return os_path.join(*seq)


# register the tag handlerpathjoin
yaml.add_constructor('!path', path)


def ToTensorLong(x: np.ndarray) -> torch.Tensor:
    '''
    Convert a number to a tensor of type long
    '''
    x = np.array(x, dtype=np.uint8)
    return torch.tensor(x, dtype=torch.long)

class ComposeTransforms:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, **kwargs):
        for transform in self.transforms:
            image, mask = transform(**kwargs)
        return image, mask

def get_train_transform():
    return {
        "input": ComposeTransforms(A.Compose([
            A.ToFloat(always_apply=True),
            ToTensorV2(),
        ]), SuperPixelTransform()),
        "target": ToTensorLong
    }


def get_val_transform():
    return {
        "input": ComposeTransforms(A.Compose([
            A.ToFloat(always_apply=True),
            ToTensorV2(),
        ]), SuperPixelTransform()),
        "target": ToTensorLong
    }


class SuperPixelTransform():
    def __init__(self, n_segments=100):
        self.n_segments = n_segments

    def generate_superpixels(self, image):
        from skimage.segmentation import slic
        segments = slic(image, n_segments=self.n_segments, multichannel=False)
        return segments

    def aggregate_superpixel_features(self, image, superpixels, p=4, r=20):
        '''
        Aggregate LBP features for each superpixel
        Capture the texture information of each superpixel

        Parameters
        ---
        image: np.ndarray
            The image to extract features from
        superpixels: np.ndarray
            The superpixel segmentation of the image
        p: int
            Number of points in a circular neighborhood
        r: int
            Radius of the circle

        Returns
        ---
        np.ndarray
            The aggregated LBP features for each superpixel
        '''
        from skimage.feature import local_binary_pattern
        lbp_img = local_binary_pattern(self.image, p, r)
        n_bins = int(lbp_img.max() + 1)  # number of bins
        features = np.zeros(
            (image.shape[0],
             image.shape[1],
             n_bins),
            dtype=np.float32)
        for label in np.unique(superpixels):
            mask = superpixels == label
            lbp_hist = np.histogram(
                lbp_img[mask],
                bins=n_bins, range=(0, n_bins),
                density=True)[0]
            for i in range(n_bins):
                features[mask, i] = lbp_hist[i]
        return features

    def aggregate_superpixel_labels(self, mask, superpixels):
        """
        Aggregate labels for each superpixel in the mask.

        Parameters
        ---
        mask: np.ndarray
            The original mask (target)
        superpixels: np.ndarray
            The superpixel segmentation of the mask

        Returns
        ---
        np.ndarray
            The aggregated labels for each superpixel
        """
        labels = np.zeros(mask.shape, dtype=np.int32)
        for label in np.unique(superpixels):
            mask_segment = superpixels == label
            labels[mask_segment] = np.bincount(mask[mask_segment]).argmax()
        return labels

    def __call__(self, image, mask):
        '''
        Compute the superpixel features and labels for the given image and mask

        Parameters
        ---
        image: np.ndarray
            The input image
        mask: np.ndarray
            The target mask

        Returns
        ---
        Tuple[np.ndarray, np.ndarray]
            The superpixel features and labels
        '''
        superpixels = self.generate_superpixels(image)
        features = self.aggregate_superpixel_features(image, superpixels)
        sp_mask = self.aggregate_superpixel_labels(mask, superpixels)
        return features, sp_mask


class UnetTransformer(Transformer):
    def __init__(self):
        super(UnetTransformer, self).__init__(
            get_train_transform(),
            get_val_transform()
        )

    def apply_train(self, input=True, **kwargs):
        if input:
            xformed = self.train_transform['input'](
                image=kwargs.get('image'),
                mask=kwargs.get('mask'))
            return xformed['image'], xformed['mask']
        else:
            return self.train_transform['target'](kwargs.get('mask'))

    def apply_val(self, input=True, **kwargs):
        if input:
            xformed = self.train_transform['input'](
                image=kwargs.get('image'),
                mask=kwargs.get('mask'))
            return xformed['image'], xformed['mask']
        else:
            return self.train_transform['target'](kwargs.get('mask'))

    def __call__(self, inputs, targets) -> List[Type[torch.Tensor]]:
        inputs, targets = self.apply_train(
            input=True, image=inputs, mask=targets)
        targets = self.apply_val(input=False, mask=targets)
        return inputs, targets


class Config(BaseConfigHandler):
    def __init__(self, file_path: str):
        super(Config, self).__init__()
        self.config = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)
        pipeline = A.Compose([
            A.ToFloat(always_apply=True),
            ToTensorV2()
        ])

        self.transform = UnetTransformer()

    def __getattr__(self, name: str) -> any:
        return self.config[name]

    def get(self, key):
        return self.config[key]

    def set(self, key, value):
        self.config[key] = value

    def update(self, key, value):
        self.set(key, value)

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    def load(self, path):
        with open(path, 'r') as f:
            for line in f:
                k, v = line.strip().split('=')
                self.set(k, v)

    def __str__(self):
        return str(self.config)
