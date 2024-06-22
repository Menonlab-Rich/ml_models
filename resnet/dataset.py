from os import listdir, path
from PIL import Image
from glob import glob
from base.dataset import GenericDataset, GenericDataLoader, GenericPredictionDataset
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import warnings
from config import Config, CONFIG_FILE_PATH
import pickle
from base.dataset import MockDataLoader



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

    def __init__(self, directory, class_labels, n_files=None, files=None):
        """
        Initialize the TargetLoader.

        Parameters:
        directory (str): Directory containing the target files.
        n_files (int): Number of files to sample (default: None).
        files (List[str]): List of files to use (optional).
        """
        self.directory = directory
        self.class_labels = class_labels
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

        # Validate the class labels. This is so we can fail early if there are any issues.
        self._validate()

    def _validate(self):
        """
        Validate the class labels.
        """
        for file in self.files:
            assert file[:
                        3] in self.class_labels, f'Invalid class label: {file[:3]}'

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
        return self.class_labels.index(file[:3])

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
            self.directory, self.class_labels, files=train_ids), self.__class__(
            self.directory, self.class_labels, files=val_ids)

class PredictionDataset(GenericPredictionDataset):
    def __init__(self, input_loader, transform=None):
        super(PredictionDataset, self).__init__(input_loader, transform)
    
    def unpack(self, inputs):
        inputs, filenames = inputs
        return inputs, filenames
    
    def __len__(self):
        return len(self.input_loader)
    
    

class ResnetDataset(GenericDataset):
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
        super(ResnetDataset, self).__init__(
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


class ResnetDataModule(LightningDataModule):
    """
    PyTorch Lightning Data Module for ResNet model.
    """

    def __init__(
            self, input_loader: InputLoader = None,
            target_loader: TargetLoader = None, prediction_loader=None,
            test_loaders=None, transforms=None, batch_size=32, n_workers=7,
            no_split=False):
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
        self.prediction_loader = prediction_loader
        if input_loader is None or target_loader is None:
            # If input_loader or target_loader is not provided, we are probably loading from a state dict
            return
        if not no_split:
            self.train_inputs, self.val_inputs = self._split_data(
                self.input_loader, 0.8)
            self.train_targets, self.val_targets = self._split_data(
                self.target_loader, 0.8)
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

    def setup(self, stage: str):
        """
        Set up the data module for different stages of training.

        Parameters:
        stage (str): Stage of training ('fit', 'test', or 'predict').
        """
        if stage in ('fit', None):
            self.train_set = ResnetDataset(
                self.train_inputs, self.train_targets, self.transforms)
            self.val_set = ResnetDataset(
                self.val_inputs, self.val_targets, self.transforms)

        if stage == 'test':
            self.test_set = self._get_test_set()

        if stage == 'predict':
            self.prediction_set = PredictionDataset(
                self.prediction_loader, self.transforms)
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
                return ResnetDataset(
                    self.val_inputs, self.val_targets, self.transforms)
        elif self.test_loaders:
            return ResnetDataset(*self.test_loaders, self.transforms)
        return None
    
    def __setattr__(self, name: str, value: any) -> None:
        """
        Set an attribute of the data module.

        Parameters:
        name (str): Name of the attribute.
        value (Any): Value to set.
        """
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
