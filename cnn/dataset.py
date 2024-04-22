import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from typing import Callable, Sequence, Any, Dict
import numpy as np


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
        self.transform = transform if transform is not None else {}
        self.inputs = input_loader()
        self.targets = target_loader()

        if len(self.inputs) != len(self.targets):
            raise ValueError(
                "Input and target datasets must have the same length.")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Sequence[torch.Tensor]:
        inp = self.inputs[index]
        target = self.targets[index]

        # Apply transformation to input if specified
        if 'input' in self.transform:
            inp = self.transform['input'](inp)

        # Apply transformation to target if specified
        if 'target' in self.transform:
            target = self.transform['target'](target)

        toTensor = ToTensorV2()
        # verify that input and target are both tensors and make them tensors if they are not
        if not torch.is_tensor(inp):
            inp = toTensor(image=inp)['image']

        if not torch.is_tensor(target):
            target = toTensor(image=target)['image']

        return inp, target

# Example usage:
# Assuming input_loader and target_loader are functions that load your data lists
# dataset = GenericDataset(input_loader=lambda: load_data('inputs'), target_loader=lambda: load_data('targets'), transform={'input': transform_input, 'target': transform_target})


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    def load_data(directory):
        # mock data loader
        return [np.zeros((10, 10, 3))]

    def transform_input(image):
        return image / 255.0

    def transform_target(target):
        return target / 255.0

    dataset = GenericDataset(
        input_loader=lambda: load_data('inputs'),
        target_loader=lambda: load_data('targets'),
        transform={'input': transform_input, 'target': transform_target})
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")
        if i == 0:
            print(f"Inputs: {inputs}")
            print(f"Targets: {targets}")
        break
