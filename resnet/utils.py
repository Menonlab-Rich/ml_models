import matplotlib.pyplot as plt
import torch
from torch.utils import data as td
import numpy as np
from dataset import GenericDataset
from typing import Any
import torch.nn as nn
from matplotlib import pyplot as plt
from typing import Dict, Literal


def save_data(dataset: GenericDataset, name: str) -> None:
    ds_obj = {
        'inputs': dataset.inputs,
        'targets': dataset.targets
    }

    torch.save(ds_obj, f'{name}.tar')


def load_data(name: str) -> tuple[Any, Any]:
    ds_obj = torch.load(f'{name}.tar')
    return ds_obj['inputs'], ds_obj['targets']


def prepare_tensor_for_plotting(tensor: torch.Tensor) -> np.ndarray:
    return tensor.permute(1, 2, 0).cpu().numpy()


def save_model(model: nn.Module, name: str) -> None:
    torch.save(model.state_dict(), f'{name}.pth')


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader,
             loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    model.to(device)
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(loader)


def plot_model_predictions(
        model: nn.Module, loader: td.DataLoader, label_map: Dict[int, str],
        num_images: int = 5, device: Literal['cpu', 'cuda'] = 'cpu'):
    """
    Plots a grid of images with their true and predicted labels from a model.

    Args:
    - model (torch.nn.Module): Trained model to predict image labels.
    - loader (torch.utils.data.DataLoader): DataLoader for the dataset.
    - label_map (dict): Dictionary mapping label indices to human-readable names.
    - num_images (int): Number of images to display.
    - device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
    - None: Displays a matplotlib plot.
    """
    model.eval()
    model.to(device)

    images, true_labels = next(iter(loader))
    if num_images > len(images):
        num_images = len(images)

    # Move images to the correct device
    images = images.to(device)

    # Predict labels
    with torch.no_grad():
        outputs = model(images)
        _, predicted_indices = torch.max(outputs, 1)

    predicted_labels = [label_map[idx.item()] for idx in predicted_indices]
    true_labels = [label_map[idx.item()] for idx in true_labels]

    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))

    for idx, ax in enumerate(axes if num_images > 1 else [axes]):
        img = images[idx].cpu().numpy().transpose((1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]

        ax.imshow(img)
        ax.axis('off')  # Turn off axis
        ax.set_title(f"True: {true_labels[idx]}\nPred: {predicted_labels[idx]}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from dataset import GenericDataset
    mock_data = np.zeros((25, 10, 10, 3))
    ds = GenericDataset(
        input_loader=lambda: mock_data,
        target_loader=lambda: mock_data,
        transform={'input': lambda x: x, 'target': lambda x: x})

    save_data(ds, 'mock_data')
    inputs, targets = load_data('mock_data')
    assert np.allclose(inputs, mock_data), 'Inputs do not match'
