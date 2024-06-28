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

    def __call__(self, image, mask):
        for transform in self.transforms:
            # recursively apply the transforms
            res = transform(image=image, mask=mask)
            if isinstance(res, tuple):
                image, mask = res
            else:
                image = res['image']
                mask = res['mask']

        return {'image': image, 'mask': mask}


def get_train_transform():
    return {
        "input": ComposeTransforms(SuperPixelTransform(),
                                   A.Compose([
                                       A.ToFloat(always_apply=True),
                                       ToTensorV2(),
                                   ])),
        "target": ToTensorLong
    }


def get_val_transform():
    return {
        "input": ComposeTransforms(SuperPixelTransform(),
                                   A.Compose([
                                       A.ToFloat(always_apply=True),
                                       ToTensorV2(),
                                   ])),
        "target": ToTensorLong
    }


import cupy as cp
import numpy as np
from cucim.skimage.feature import local_binary_pattern
from cucim.skimage.segmentation import slic

class SuperPixelTransform():
    def __init__(self, n_segments=100):
        self.n_segments = n_segments

    def generate_superpixels(self, image):
        segments = slic(image, n_segments=self.n_segments, channel_axis=None)
        return segments

    def aggregate_superpixel_features(self, image, superpixels, p=4, r=20):
        '''
        Aggregate LBP features for each superpixel
        Capture the texture information of each superpixel

        Parameters
        ---
        image: cp.ndarray
            The image to extract features from
        superpixels: cp.ndarray
            The superpixel segmentation of the image
        p: int
            Number of points in a circular neighborhood
        r: int
            Radius of the circle

        Returns
        ---
        cp.ndarray
            The aggregated LBP features for each superpixel
        '''
        lbp_img = local_binary_pattern(image, p, r)
        n_bins = int(lbp_img.max() + 1)  # number of bins
        features = cp.zeros((image.shape[0], image.shape[1], n_bins), dtype=cp.float32)
        
        for label in cp.unique(superpixels):
            mask = superpixels == label
            lbp_hist = cp.histogram(lbp_img[mask], bins=n_bins, range=(0, n_bins), density=True)[0]
            for i in range(n_bins):
                features[mask, i] = lbp_hist[i]
        
        return features

    def process_batch(self, images):
        '''
        Process a batch of images to generate superpixels and aggregate features.

        Parameters
        ---
        images: List[np.ndarray]
            List of images to process.

        Returns
        ---
        List[cp.ndarray]
            List of aggregated superpixel features for each image.
        '''
        batch_features = []
        
        for image in images:
            image_gpu = cp.asarray(image)  # Transfer image to GPU
            superpixels = self.generate_superpixels(image_gpu)
            features = self.aggregate_superpixel_features(image_gpu, superpixels)
            batch_features.append(features)
        
        return batch_features

# Example usage
if __name__ == "__main__":
    import cv2  # Optional: for image loading

    # Load images (example with two grayscale images)
    image1 = cv2.imread('path_to_image1', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('path_to_image2', cv2.IMREAD_GRAYSCALE)

    # Initialize the transform
    transform = SuperPixelTransform(n_segments=100)

    # Process a batch of images
    images = [image1, image2]
    batch_features = transform.process_batch(images)

    # Convert results back to CPU and print the shapes
    for features in batch_features:
        features_cpu = cp.asnumpy(features)
        print(features_cpu.shape)



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
