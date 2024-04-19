import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from typing import Callable, Sequence, Any, Dict
import numpy as np
from typing import List
from torch.utils.data import Subset
import copy


class TransformSubset():
    '''
    A Subset that applies a transformation to the data.
    '''

    def __init__(
            self, subset: Subset,
            transform: Dict[str, Callable[[np.ndarray],
                                          Any]] = None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset) 

    def __getitem__(self, idx):
        # Get the item from the main dataset

        # Apply transformation if specified
        if self.transform:
            x, y = self.subset.dataset[self.subset.indices[idx]]
            if 'input' in self.transform:
                x = self.transform['input'](x)
            if 'target' in self.transform:
                y = self.transform['target'](y)

            if not torch.is_tensor(x):
                x = ToTensorV2()(image=x)['image']
            if not torch.is_tensor(y):
                y = ToTensorV2()(image=y)['image']
        else:
            x, y = self.subset[idx]
        return x, y


class GenericDataset(data.Dataset):
    '''
    A generic dataset class that can be used with any input and target data.
    Supports custom transformations for input and target data.
    '''

    def __init__(
            self, input_loader: Callable[[],
                                         Sequence[Any]],
            target_loader: Callable[[],
                                    Sequence[Any]],
            transform: Dict[str, Callable[[np.ndarray], Any]] = None):
        '''
        Create a new GenericDataset object.

        Parameters:
        ----------
        input_loader: A function that loads the input data. Must return a sequence of inputs.
        target_loader: A function that loads the target data. Must return a sequence of targets.
        transform: A dictionary containing transformations for input and target data.
        '''
        self.transform = self._standardize_transform(transform)
        self.inputs = input_loader()
        self.targets = target_loader()

        if len(self.inputs) != len(self.targets):
            raise ValueError(
                "Input and target datasets must have the same length.")


    def _standardize_transform(self,
                               transform: Dict
                               [str, Callable[[np.ndarray],
                                              Any]]) -> Dict[str,
                                                             Callable[[np.ndarray],
                                                                      Any]]:
        '''
        Standardize the transform dictionary to be of the form
        {'train': {'input': lambda x: x, 'target': lambda x: x}, 'val': {'input': lambda x: x, 'target': lambda x: x}}
        '''
        default_transform = {
            'train': {'input': lambda x: x, 'target': lambda x: x},
            'val': {'input': lambda x: x, 'target': lambda x: x}}
        if transform is None:
            return default_transform
        if 'train' not in transform:
            transform['train'] = default_transform['train']
        if 'val' not in transform:
            transform['val'] = default_transform['val']
        if 'input' not in transform['train']:
            transform['train']['input'] = default_transform['train']['input']
        if 'target' not in transform['train']:
            transform['train']['target'] = default_transform['train']['target']
        if 'input' not in transform['val']:
            transform['val']['input'] = default_transform['val']['input']
        if 'target' not in transform['val']:
            transform['val']['target'] = default_transform['val']['target']
        return transform

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Sequence[torch.Tensor]:
        inp = self.inputs[index]
        target = self.targets[index]

        if 'train' in self.transform:
            train_transforms = self.transform['train']
            # Apply transformation to input if specified
            if 'input' in train_transforms:
                inp = train_transforms['input'](inp)

            # Apply transformation to target if specified
            if 'target' in train_transforms:
                target = train_transforms['target'](target)

        toTensor = ToTensorV2()
        # verify that input and target are both tensors and make them tensors if they are not
        if not torch.is_tensor(inp):
            inp = toTensor(image=inp)['image']

        if not torch.is_tensor(target):
            target = toTensor(image=target)['image']

        return inp, target

    def split(self, train_ratio: float) -> List[data.Subset]:
        train_size = int(train_ratio * len(self))
        val_size = len(self) - train_size
        train_set, val_set = data.random_split(self, [train_size, val_size])
        return TransformSubset(
            train_set, self.transform['train']), TransformSubset(
            val_set, self.transform['val'])

# Example usage:
# Assuming input_loader and target_loader are functions that load your data lists
# dataset = GenericDataset(input_loader=lambda: load_data('inputs'), target_loader=lambda: load_data('targets'), transform={'input': transform_input, 'target': transform_target})


if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader

    def load_data(directory):
        # mock data loader
        return [np.zeros((10, 10, 3)), np.zeros((10, 10, 3)), np.zeros((10, 10, 3))]

    def transform_input(image):
        return image / 255.0

    def transform_target(target):
        return target / 255.0

    def transform_val_input(image):
        return np.where(image < 1, 1, 0)

    def transform_val_target(target):
        return np.where(target < 1, 1, 0)

    dataset = GenericDataset(
        input_loader=lambda: load_data('inputs'),
        target_loader=lambda: load_data('targets'),
        transform={
            'train':
            {'input': transform_input, 'target': transform_target},
            'val':
            {'input': transform_val_input,
             'target': transform_val_target}})
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")
        if i == 0:
            print(f"Inputs: {inputs}")
            print(f"Targets: {targets}")
        break

    train_set, val_set = dataset.split(0.8)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    for i, (inputs, targets) in enumerate(train_loader):
        print(f"Train Batch {i}:")
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")
        if i == 0:
            print(f"Inputs: {inputs}")
            print(f"Targets: {targets}")
        break
