import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from typing import Callable, Sequence, Any, Dict
import numpy as np
from typing import List, Tuple


class GenericDataLoader():
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def get_ids(self, i=None):
        pass

    def __iter__(self):
        pass

    def split(self, train_ratio: float, seed=16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split dataset IDs into training and validation sets based on the given train ratio.

        Parameters:
            train_ratio (float): The proportion of the dataset to include in the training set.
            seed (int): Seed for the random number generator to ensure reproducibility.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of training IDs and validation IDs.
        """
        if not seed or seed == 0:
            raise ValueError(
                "Seed must be a non-zero integer to ensure reproducibility.")

        ids = self.get_ids()  # Get all the ids
        train_size = int(train_ratio * len(ids))

        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Randomly sample the train IDs without replacement
        train_ids = np.random.choice(ids, train_size, replace=False)

        # Get the remaining IDs for the validation set
        val_ids = np.setdiff1d(ids, train_ids)

        return train_ids, val_ids


class Transformer:
    def __init__(self, train_transform, val_transform):
        self.train_transform = train_transform
        self.val_transform = val_transform

    def apply_train(self, **kwargs):
        raise NotImplementedError
    
    def apply_val(self, x):
        raise NotImplementedError


class GenericDataset(data.Dataset):
    '''
    A generic dataset class that can be used with any input and target data.
    Supports custom transformations for input and target data.
    '''

    def __init__(
            self, input_loader: GenericDataLoader,
            target_loader: GenericDataLoader,
            transform: Transformer):
        '''
        Create a new GenericDataset object.

        Parameters:
        ----------
        input_loader: A function that loads the input data. Must return a sequence of inputs.
        target_loader: A function that loads the target data. Must return a sequence of targets.
        transform: A dictionary containing transformations for input and target data.
        '''

        self.input_loader = input_loader
        self.target_loader = target_loader
        self.input_identifiers = input_loader.get_ids()
        self.target_identifiers = target_loader.get_ids()
        self.transform = transform

        if len(self.input_loader) != len(self.target_loader):
            raise ValueError(
                "Input and target datasets must have the same length.")

    def __len__(self) -> int:
        return len(self.input_loader)

    def unpack(self, inputs, targets):
        '''
        Unpack the input and target values if required

        Override this method if you need to unpack the inputs and targets in a specific way.
        The returned values should be in the order: inputs, targets, *rest
        The *rest values will be returned as additional values when the dataset is accessed.
        This is because we may need to return additional values along with the inputs and targets but we don't want to transform them.
        Ensuring that the *rest values are the last values returned will allow us to identify the transformed and non-transformed values.
        '''
        return inputs, targets

    def __getitem__(self, index: int) -> Sequence[torch.Tensor]:
        inp = self.input_loader[index]
        target = self.target_loader[index]
        inp, target, *rest = self.unpack(inp, target)
        inp, target = self.transform(inp, target)

        return inp, target, rest
