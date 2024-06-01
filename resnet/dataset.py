from os import listdir, path
from PIL import Image
from glob import glob
from base.dataset import GenericDataset, GenericDataLoader
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import warnings
from config import Config, CONFIG_FILE_PATH
import pickle


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
            return self.files[i:i+batch_size - 1]
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
        return np.array(Image.open(path.join(self.directory, file)))

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


class ResnetDataModule(LightningDataModule):
    """
    PyTorch Lightning Data Module for ResNet model.
    """

    def __init__(
            self, input_loader: InputLoader = None,
            target_loader: TargetLoader = None, prediction_loaders=None,
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
        super(ResnetDataModule, self).__init__()
        assert input_loader is not None, 'Input loader must be provided'
        assert target_loader is not None, 'Target loader must be provided'

        # Deserialize loaders if they are bytes
        if isinstance(input_loader, bytes):
            input_loader = pickle.loads(input_loader)
        if isinstance(target_loader, bytes):
            target_loader = pickle.loads(target_loader)

        self.test_loaders = test_loaders

        if not no_split:
            self.train_inputs, self.val_inputs = input_loader.split(0.8)
            self.train_targets, self.val_targets = target_loader.split(0.8)
        else:
            self.train_inputs = input_loader
            self.val_inputs = input_loader
            self.train_targets = target_loader
            self.val_targets = target_loader

        self.transforms = transforms
        self.batch_size = batch_size
        self.prediction_loader = prediction_loaders
        self.n_workers = n_workers
        self.test_loaders = test_loaders

        hparams = {
            "input_loader": pickle.dumps(input_loader),
            "target_loader": pickle.dumps(target_loader),
            "prediction_loader": pickle.dumps(prediction_loaders)
            if prediction_loaders is not None else None,
            "test_loaders": pickle.dumps(test_loaders)
            if test_loaders is not None else None, "transforms": pickle.dumps(
                transforms),
            "batch_size": batch_size, "n_workers": n_workers}
        self.save_hyperparameters(hparams)

    def setup(self, stage: str):
        """
        Set up the data module for different stages of training.

        Parameters:
        stage (str): Stage of training ('fit', 'test', or 'predict').
        """
        if stage == 'fit' or stage is None:
            self.train_set = ResnetDataset(
                self.train_inputs, self.train_targets, self.transforms)
            self.val_set = ResnetDataset(
                self.val_inputs, self.val_targets, self.transforms)

        if stage == 'test':
            if self.test_loaders is not None and not isinstance(
                    self.test_loaders, str):
                self.test_set = ResnetDataset(
                    self.test_loaders[0],
                    self.test_loaders[1],
                    self.transforms)
            elif isinstance(self.test_loaders, str):
                if self.test_loaders.lower() == 'validation':
                    self.test_set = ResnetDataset(
                        self.val_inputs, self.val_targets, self.transforms)
            else:
                self.test_set = None

        if stage == 'predict':
            if self.prediction_loaders is not None:
                self.prediction_set = ResnetDataset(
                    self.prediction_loader[0],
                    self.prediction_loader[1],
                    self.transforms)
            else:
                self.prediction_set = None

    def state_dict(self):
        """
        Save the state of the data module.

        Returns:
        dict: State dictionary.
        """
        try:
            return {
                'batch_size': self.batch_size,
                'n_workers': self.n_workers
            }
        except Exception as e:
            warnings.warn(f'Error in saving state dict: {e}')

    def load_state_dict(self, state_dict):
        """
        Load the state of the data module.

        Parameters:
        state_dict (dict): State dictionary.
        """
        try:
            inputs = [
                path.join(config.data_dir, path.basename(file))
                for file in state_dict['train_inputs']]
            targets = [
                path.join(path.basename(file))
                for file in state_dict['train_targets']]
            val_inputs = [
                path.join(config.data_dir, path.basename(file))
                for file in state_dict['val_inputs']]
            val_targets = [
                path.join(path.basename(file))
                for file in state_dict['val_targets']]
            self.train_inputs = InputLoader(files=inputs)
            self.val_inputs = InputLoader(files=val_inputs)
            self.train_targets = TargetLoader(files=targets)
            self.val_targets = TargetLoader(files=val_targets)
            self.batch_size = state_dict['batch_size']
            self.n_workers = state_dict['n_workers']
        except Exception as e:
            warnings.warn(f'Error in loading state dict: {e}')

    def train_dataloader(self):
        """
        Get the DataLoader for training data.

        Returns:
        DataLoader: DataLoader for training data.
        """
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_workers, persistent_workers=True)

    def val_dataloader(self):
        """
        Get the DataLoader for validation data.

        Returns:
        DataLoader: DataLoader for validation data.
        """
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_workers, persistent_workers=True)

    def test_dataloader(self):
        """
        Get the DataLoader for test data.

        Returns:
        DataLoader: DataLoader for test data, or None if test set is not defined.
        """
        if self.test_set is not None:
            return DataLoader(
                self.test_set, batch_size=self.batch_size, shuffle=False,
                num_workers=self.n_workers, persistent_workers=True)

    def predict_dataloader(self):
        """
        Get the DataLoader for prediction data.

        Returns:
        DataLoader: DataLoader for prediction data, or validation DataLoader if prediction set is not defined.
        """
        if self.prediction_set is not None:
            return DataLoader(
                self.prediction_set, batch_size=self.batch_size, shuffle=False,
                num_workers=self.n_workers, persistent_workers=True)
        else:
            warnings.warn('No prediction set provided. Using validation set.')
            return self.val_dataloader()
