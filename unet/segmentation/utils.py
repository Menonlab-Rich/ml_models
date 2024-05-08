import matplotlib.pyplot as plt
import torch
from torch.utils import data as td
import numpy as np
from base.dataset import GenericDataset
from typing import Sequence, Tuple
import torch.nn as nn
from matplotlib import pyplot as plt
from typing import Dict, Literal, List
from base.utilities import BaseUtilities
import torch.nn.functional as F
from base.loss import JaccardLoss


class utils(BaseUtilities):
    @classmethod
    def save_data(self, dataset: GenericDataset, name: str):
        ds_obj = {
            'inputs': dataset.input_loader.get_ids(),
            'targets': dataset.target_loader.get_ids()
        }

        torch.save(ds_obj, f'{name}.tar')

    @classmethod
    def load_data(self, name: str):
        ds_obj = torch.load(f'{name}.tar')
        return ds_obj['inputs'], ds_obj['targets']

    @classmethod
    def prepare_img_tensors_for_plotting(
            self, *tensors: torch.Tensor) -> np.ndarray:

        imgs = [t.permute(1, 2, 0).cpu().detach().numpy() for t in tensors]
        return imgs if len(imgs) > 1 else imgs[0]

    @classmethod
    def save_checkpoint(
        self, model: nn.Module, optimizer: torch.optim.Optimizer,
            epoch: int, name: str):
        with open(name, 'wb') as f:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f)

    @classmethod
    def load_checkpoint(
            self, model: nn.Module, optimizer: torch.optim.Optimizer, device,
            filename: str) -> int:
        checkpoint = torch.load(filename, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    @classmethod
    def evaluate(self, model: nn.Module, loader: torch.utils.data.DataLoader,
                 loss_fn: nn.Module, device: torch.device) -> float:
        model.eval()
        model.to(device)
        running_loss = 0.0
        loader.dataset.return_identifiers = False  # Ensure we don't return identifiers
        total = 0
        total_correct = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_loss += loss.item()
                total += targets.size(0)
                total_correct += (torch.argmax(outputs, 1)
                                  == targets).sum().item()
        # save the total number of correct predictions

        return running_loss / len(loader)

    @classmethod
    def plot_model_predictions(
            self, model: nn.Module, loader: torch.utils.data.DataLoader,
            label_map: Dict[int, str],
            num_images: int = 5, device: Literal['cpu', 'cuda'] = 'cpu',
            output_path: str = None):
        """
        Plots a grid of images with their true and predicted labels from a model.

        Args:
        - model (torch.nn.Module): Trained model to predict image labels.
        - loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        - label_map (dict): Dictionary mapping label indices to human-readable names.
        - num_images (int): Number of images to display.
        - device (str): Device to perform computations on ('cpu' or 'cuda').
        - output_path (str): Path to save the plot.

        Returns:
        - None: Displays a matplotlib plot.
        """
        from os import path
        model.eval().to(device)
        loader.dataset.return_identifiers = True
        images, labels, filenames, _ = next(iter(loader))
        if num_images > len(images):
            num_images = len(images)
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            predicted_indices = torch.argmax(outputs, 1)
        predicted_labels = [label_map[idx.item()] for idx in predicted_indices]
        true_labels = [label_map[idx.item()] for idx in labels]
        fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
        for idx, ax in enumerate(axes if num_images > 1 else [axes]):
            img = images[idx].cpu().numpy().transpose((1, 2, 0))
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min())
            ax.imshow(img)
            ax.axis('off')  # Turn off axis
            ax.set_title(
                f"Filename: {path.basename(filenames[idx])}\nTrue: {true_labels[idx]}\nPred: {predicted_labels[idx]}")
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()

    @classmethod
    def prepare_label_tensors_for_plotting(cls, colors: Sequence[Tuple[int]], *labels: torch.Tensor) -> List[np.ndarray]:
        """
        Convert list of label tensors to RGB color-coded arrays.

        Args:
        labels: Variable-length list of torch tensors, each containing predicted class indices.

        Returns:
        A list of numpy arrays, each corresponding to the input tensors color-coded.
        """
        def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
            # Create an empty RGB array
            height, width = tensor.shape
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

            for label_index, color in enumerate(colors):
                mask = tensor == label_index
                rgb_image[mask] = color

            return rgb_image

        # Apply `tensor_to_rgb` for each label tensor
        imgs = [tensor_to_rgb(label.cpu().detach().numpy()) for label in labels]
        return imgs if len(imgs) > 1 else imgs[0]


class Evaluator:
    def __init__(self, model: nn.Module, loader: td.DataLoader,
                 loss_fn: nn.Module, device: torch.device, config: dict,
                 weights=None):
        self.model = model
        self.loader = loader
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.percent_correct_per_epoch = []
        self.losses_per_epoch = []
        self.weights = [
            1.0] * config['num_classes'] if weights is None else weights
        self.weights = torch.tensor(
            self.weights, dtype=torch.float32).to(device)
        self.jaccard = JaccardLoss(len(self.weights), self.weights)
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        self.total_predictions = 0

    def update(self, loss: torch.Tensor, accuracy: torch.Tensor, batch_size: int):
        self.running_loss += loss.item()
        self.running_accuracy += accuracy.item()
        self.total_predictions += batch_size

    def evaluate(self) -> float:
        avg_loss = self.running_loss / self.total_predictions
        avg_accuracy = self.running_accuracy / self.total_predictions
        self.percent_correct_per_epoch.append(avg_accuracy)
        self.losses_per_epoch.append(avg_loss)
        self.running_loss = 0.0
        self.running_accuracy = 0.0
    

    def accuracy(self, predictions, ground_truths) -> float:
        '''
        Return the accuracy of the model as a Jaccard Index.
        '''
        self.model.eval()
        with torch.no_grad():
            jaccard_score = self.jaccard.forward(predictions, ground_truths).item() + 1. # Add 1 to convert to accuracy
        self.model.train()
        return jaccard_score * 100

    def plot(self, metrics='both', output_path: str = None):
        fig, ax = plt.subplots(
            2 if metrics == 'both' else 1, 1,
            figsize=(10, 5 if metrics == 'both' else 10))

        if metrics in ('both', 'accuracy'):
            ax_acc = ax[0] if metrics == 'both' else ax
            ax_acc.plot(self.percent_correct_per_epoch)
            ax_acc.set_title('Percent Correct per Epoch')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Percent Correct')

        if metrics in ('both', 'loss'):
            ax_loss = ax[1] if metrics == 'both' else ax
            ax_loss.plot(self.losses_per_epoch)
            ax_loss.set_title('Loss per Epoch')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
