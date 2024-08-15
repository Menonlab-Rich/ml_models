from base.dataset import GenericDataLoader, GenericDataset, DynamicDataset
from pytorch_lightning import LightningDataModule
from PIL import Image
import numpy as np
import os
from os import path, listdir
from glob import glob
import pickle
import warnings
from torch.utils.data import DataLoader
from numpy.lib.stride_tricks import view_as_blocks
from typing import Iterable, Tuple


# Example implementation of the generate function
def patch_generator(input_image: np.ndarray, target_image: np.ndarray, patch_size: int | None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate patches from the input and target images.

    Parameters:
    input_image (np.ndarray): The input image.
    target_image (np.ndarray): The target image/mask.
    patch_size (int): The size of the patches to generate.

    Returns:
    Iterable[Tuple[np.ndarray, np.ndarray]]: An iterable of (input_patch, target_patch) tuples.
    """
    if patch_size is None:
        yield input_image, target_image
    # Crop target to non-zero region
    nz = np.nonzero(target_image)
    min_row, max_row = np.min(nz[0]), np.max(nz[0])
    min_col, max_col = np.min(nz[1]), np.max(nz[1])

    # Ensure the cropped region is divisible by the patch size
    min_row = (min_row // patch_size) * patch_size
    max_row = ((max_row // patch_size) + 1) * patch_size
    min_col = (min_col // patch_size) * patch_size
    max_col = ((max_col // patch_size) + 1) * patch_size

    cropped_target = target_image[min_row:max_row, min_col:max_col]
    cropped_input = input_image[min_row:max_row, min_col:max_col]

    # Split into patches
    target_patches = view_as_blocks(cropped_target, block_shape=(patch_size, patch_size))
    input_patches = view_as_blocks(cropped_input, block_shape=(patch_size, patch_size))

    for i in range(target_patches.shape[0]):
        for j in range(target_patches.shape[1]):
            target_patch = target_patches[i, j].reshape(patch_size, patch_size)
            input_patch = input_patches[i, j].reshape(patch_size, patch_size)
            # Filter out patches with less than 1% non-zero pixels in the target
            if np.count_nonzero(target_patch) >= 0.01 * np.prod(target_patch.shape):
                yield input_patch, target_patch

class InputLoader(GenericDataLoader):
    """
    Loader for input files (images) to be used as input to the model.
    """

    def __init__(self, directory, n_files=None, files=None):
        """
        Initialize the InputLoader.

        Parameters:
        directory (str): Directory containing the input files.
        n_files (int): Number of files to sample (default: None).
        files (List[str]): List of files to use (optional).
        """

        self.directory = directory
        if files is not None:
            self.files = files
        elif not path.isdir(directory):
            # Handle glob pattern
            self.files = sorted(glob(directory))
            self.directory = path.dirname(directory)
        else:
            self.files = sorted(
                [f for f in listdir(directory) if f.endswith('.tif')])
        if n_files is not None:
            self.files = self.files[:n_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self._read(self.files[idx])

    def __iter__(self):
        for file in self.files:
            yield self._read(file)

    def get_ids(self, i=None, batch_size=1):
        """
        Get the file names for the given index and batch size.

        Parameters:
        i (int): Index (default: None).
        batch_size (int): Batch size (default: 1).

        Returns:
        List[str]: List of file names.
        """
        if i is not None:
            return self.files[i:i+batch_size]
        return self.files

    def _read(self, file):
        """
        Read the file and return it as a numpy array.

        Parameters:
        file (str): File name.

        Returns:
        np.ndarray: Numpy array of the image.
        """
        file = path.basename(file)
        return np.array(Image.open(path.join(self.directory, file))), file

    def post_split(self, train_ids, val_ids):
        """
        Split the loader into training and validation sets.

        Parameters:
        train_ids (List[str]): List of training file names.
        val_ids (List[str]): List of validation file names.

        Returns:
        Tuple[InputLoader, InputLoader]: Training and validation loaders.
        """
        return InputLoader(
            self.directory, files=train_ids), InputLoader(
            self.directory, files=val_ids)


class TargetLoader(GenericDataLoader):
    """
    Loader for target files (class labels) corresponding to the input files.
    """

    def __init__(self, directory, n_files=None, files=None):
        """
        Initialize the TargetLoader.

        Parameters:
        directory (str): Directory containing the target files.
        n_files (int): Number of files to sample (default: None).
        files (List[str]): List of files to use (optional).
        """
        self.directory = directory
        if files is not None:
            self.files = files
        elif not path.isdir(directory):
            # Handle glob pattern
            self.files = sorted(glob(directory))
            self.directory = path.dirname(directory)
        else:
            self.files = sorted(
                [f for f in listdir(directory)
                 if f.endswith('.npy') or f.endswith('.npz')])
        if n_files is not None:
            self.files = self.files[:n_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self._read(self.files[idx])

    def __iter__(self):
        for file in self.files:
            yield self._read(file)

    def get_ids(self, i=None, batch_size=1):
        """
        Get the file names for the given index and batch size.

        Parameters:
        i (int): Index (default: None).
        batch_size (int): Batch size (default: 1).

        Returns:
        List[str]: List of file names.
        """
        if i is not None:
            return self.files[i:i+batch_size - 1]
        return self.files

    def _read(self, file: str) -> int:
        """
        Read the class of the file based on the first three characters of the file name.

        Parameters:
        file (str): File name.

        Returns:
        int: Class index.
        """
        file = path.basename(file)
        if file.endswith('.npy'):
            return np.load(path.join(self.directory, file))
        elif file.endswith('.npz'):
            data = np.load(path.join(self.directory, file))
            if 'mask' in data:
                return data['mask']
            return data['arr_0']
        else:
            raise ValueError(f'Invalid file format: {file}')

    def post_split(self, train_ids, val_ids):
        """
        Split the loader into training and validation sets.

        Parameters:
        train_ids (List[str]): List of training file names.
        val_ids (List[str]): List of validation file names.

        Returns:
        Tuple[TargetLoader, TargetLoader]: Training and validation loaders.
        """
        return self.__class__(
            self.directory, files=train_ids), self.__class__(
            self.directory, files=val_ids)

class UnetDataset(GenericDataset):
    """
    Dataset class for ResNet model.
    """

    def __init__(self, input_loader, target_loader, transform=None):
        """
        Initialize the ResnetDataset.

        Parameters:
        input_loader (InputLoader): Loader for input files.
        target_loader (TargetLoader): Loader for target files.
        transform (callable): Transformation function (default: None).
        """
        super(UnetDataset, self).__init__(
            input_loader, target_loader, transform)

    def unpack(self, inputs, targets):
        """
        Unpack the data into input and target.

        Parameters:
        data (Tuple[np.ndarray, int]): Tuple containing input and target.

        Returns:
        Tuple[np.ndarray, int]: Unpacked input and target.
        """
        inputs, filenames = inputs
        return inputs, targets, filenames


class UNetDataModule(LightningDataModule):
    """
    PyTorch Lightning Data Module for ResNet model.
    """

    def __init__(
            self, input_loader: InputLoader = None,
            target_loader: TargetLoader = None, prediction_loader=None,
            test_loaders=None, transforms=None, batch_size=32, n_workers=7,
            split_ratio=0.8, no_split=False, patch_size=None):
        """
        Initialize the ResnetDataModule.

        Parameters:
        input_loader (InputLoader): Loader for input files.
        target_loader (TargetLoader): Loader for target files.
        prediction_loaders (list): List of loaders for prediction (default: None).
        test_loaders (list): List of loaders for testing (default: None).
        transforms (callable): Transformation function (default: None).
        batch_size (int): Batch size (default: 32).
        n_workers (int): Number of workers (default: 7).
        no_split (bool): If True, do not split the data (default: False).
        """
        super().__init__()
        self.input_loader = self._load_if_bytes(input_loader)
        self.target_loader = self._load_if_bytes(target_loader)
        self.test_loaders = test_loaders
        self.transforms = transforms
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.patch_size = patch_size
        self.prediction_loader = prediction_loader
        if input_loader is None or target_loader is None:
            # If input_loader or target_loader is not provided, we are probably loading from a state dict
            return
        if not no_split:
            self.train_inputs, self.val_inputs = self._split_data(
                self.input_loader, split_ratio)
            self.train_targets, self.val_targets = self._split_data(
                self.target_loader, split_ratio)
        else:
            self.train_inputs = self.val_inputs = self.input_loader
            self.train_targets = self.val_targets = self.target_loader

        self.save_hyperparameters({
            "batch_size": batch_size, "n_workers": n_workers,
        })
    
    def _split_data(self, loader:GenericDataLoader, split_ratio:float):
        """
        Split the data into training and validation sets.

        Parameters:
        data (list): List of data.
        split_ratio (float): Ratio to split the data.

        Returns:
        Tuple[list, list]: Training and validation sets.
        """
        return loader.split(split_ratio)

    def _load_if_bytes(self, obj):
        if isinstance(obj, bytes) and len(obj) > 0:
            return pickle.loads(obj)
        elif isinstance(obj, bytes):
            return None
        return obj
    
    def _wrap_dataset(self, dataset):
        '''
        Wrap the dataset with a patch generator.
        This way, the patch generation is done on the fly during training.
        '''
        return DynamicDataset(dataset, lambda x, y, *_: patch_generator(x, y, self.patch_size))

    def setup(self, stage: str):
        """
        Set up the data module for different stages of training.

        Parameters:
        stage (str): Stage of training ('fit', 'test', or 'predict').
        """
        if stage in ('fit', None):
            self.train_set = UnetDataset(
                self.train_inputs, self.train_targets, self.transforms)
            self.val_set = UnetDataset(
                self.val_inputs, self.val_targets, self.transforms)

        if stage == 'test':
            self.test_set = self._get_test_set()

        if stage == 'predict':
            self.prediction_set = UnetDataset(
                self.prediction_loader, None, self.transforms)
    def _get_test_set(self):
        if isinstance(self.test_loaders, str):
            if self.test_loaders.lower() == 'validation':
                if getattr(self, 'val_inputs', None) is None:
                    warnings.warn('Validation set not found. Trying to load from input_loader state dict.')
                    try:
                        self.val_inputs = self.val_set.input_loader
                        self.val_targets = self.val_set.target_loader
                    except AttributeError:
                        raise ValueError('Validation set not found.')
                return UnetDataset(
                    self.val_inputs, self.val_targets, self.transforms)
        elif self.test_loaders:
            return UnetDataset(*self.test_loaders, self.transforms)
        return None
    
    def __setattr__(self, name: str, value: any) -> None:
        """
        Set an attribute of the data module.

        Parameters:
        name (str): Name of the attribute.
        value (Any): Value to set.
        """

        # Prevent overwriting existing attributes
        # Required when loading from state dict
        if name in ('input_loader', 'target_loader', 'prediction_loader', 'test_loaders'):
            if getattr(self, name, None) is not None:
                warnings.warn(f'Refusing to set {name} as it is already set.')
                return
        super().__setattr__(name, value)

    def state_dict(self):
        """
        Save the state of the data module.

        Returns:
        dict: State dictionary.
        """
        return {
            'batch_size': self.batch_size, 
            'n_workers': self.n_workers,
            'input_loader': pickle.dumps(self.input_loader),
            'target_loader': pickle.dumps(self.target_loader),
            'prediction_loader': pickle.dumps(self.prediction_loader) if self.prediction_loader else b'',
            'test_loaders': pickle.dumps(self.test_loaders) if self.test_loaders else b'',
            'transforms': pickle.dumps(self.transforms) if self.transforms else b'',
        }

    def load_state_dict(self, state_dict):
        """
        Load the state of the data module.

        Parameters:
        state_dict (dict): State dictionary.
        """
        try:
            self.batch_size = state_dict['batch_size']
            self.n_workers = state_dict['n_workers']
            self.input_loader = self._load_if_bytes(state_dict['input_loader'])
            self.target_loader = self._load_if_bytes(state_dict['target_loader'])
            self.prediction_loader = self._load_if_bytes(state_dict['prediction_loader'])
            self.test_loaders = self._load_if_bytes(state_dict['test_loaders'])
            self.transforms = self._load_if_bytes(state_dict['transforms'])
            if (not self.input_loader or not self.target_loader) and (not self.prediction_loader and not self.test_loaders):
                raise ValueError('Input or target loader not found.')
        except KeyError as e:
            warnings.warn(f'Error in loading state dict: {e}')
        self.save_hyperparameters({
            "batch_size": self.batch_size, "n_workers": self.n_workers,
        })
        self.train_inputs, self.val_inputs = self._split_data(
            self.input_loader, 0.8)
        self.train_targets, self.val_targets = self._split_data(
            self.target_loader, 0.8)
    def train_dataloader(self):
        """
        Get the DataLoader for training data.

        Returns:
        DataLoader: DataLoader for training data.
        """
        return self._create_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        """
        Get the DataLoader for validation data.

        Returns:
        DataLoader: DataLoader for validation data.
        """
        return self._create_dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        """
        Get the DataLoader for test data.

        Returns:
        DataLoader: DataLoader for test data, or None if test set is not defined.
        """
        return self._create_dataloader(
            self.test_set, shuffle=False) if self.test_set else None

    def predict_dataloader(self):
        """
        Get the DataLoader for prediction data.

        Returns:
        DataLoader: DataLoader for prediction data, or validation DataLoader if prediction set is not defined.
        """
        if self.prediction_set:
            return self._create_dataloader(self.prediction_set, shuffle=False)
        warnings.warn('No prediction set provided. Using validation set.')
        return self.val_dataloader()

    def _create_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.n_workers, persistent_workers=True)
