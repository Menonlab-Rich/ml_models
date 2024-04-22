import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torch import nn
import config
from ml_models.cnn.dataset import GenericDataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
import os
from typing import Tuple, Sequence


def save_model(
        model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
        loss: float) -> None:
    '''
    Save the model to disk
    '''
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(
        config.checkpoint_dir, f'checkpoint_{epoch}.tar'))
    config.logger.info(
        f"Saved model to {os.path.join(config.checkpoint_dir, f'checkpoint_{epoch}.tar')}")


def load_model(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> Tuple[nn.Module, torch.optim.Optimizer, int, float]:
    '''
    Load the model from disk
    '''
    checkpoint = torch.load(os.path.join(
        config.checkpoint_dir, f'checkpoint_{epoch}.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    config._logger.info(
        f"Loaded model from {os.path.join(config.checkpoint_dir, f'checkpoint_{epoch}.tar')}")
    return model, optimizer, epoch, loss


def save_dataset(dataset: GenericDataset) -> None:
    inputs = dataset.inputs
    targets = dataset.targets
    torch.save(inputs, os.path.join(config.dset_dir, 'inputs.tar'))
    torch.save(targets, os.path.join(config.dset_dir, 'targets.tar'))


def load_dataset() -> GenericDataset:
    input_path = os.path.join(config.dset_dir, 'inputs.tar')
    target_path = os.path.join(config.dset_dir, 'targets.tar')
    inputs = torch.load(input_path)
    targets = torch.load(target_path)
    dataset = GenericDataset(
        lambda: inputs, lambda: targets, transform=config.transforms)
    return dataset


def predict_classes(logits: torch.Tensor) -> torch.Tensor:
    '''
    Given a tensor of logits, return the class with the highest probability
    '''
    probability = torch.nn.functional.softmax(logits, dim=1)
    return torch.argmax(probability, dim=1)


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def plot_results(
        model: nn.Module, val_loader: DataLoader, epoch: int, num_images=3) -> None:
    '''
    Plot the results of the model on the validation set
    '''
    model.eval()
    to_pil = ToPILImage()
    fig, ax = plt.subplots(3, num_images, figsize=(15, 5))

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i == num_images:
                break
            inputs, targets = inputs.to(
                config.device), targets.to(
                config.device)
            outputs = model(inputs)
            outputs = predict_classes(outputs)
            for j in range(inputs.shape[0]):
                input_img = to_pil(inputs[j].cpu())
                target_img = to_pil(targets[j].cpu())
                output_img = to_pil(outputs[j].cpu())
                ax[0, j].imshow(input_img)
                ax[0, j].set_title("Input")

                ax[1, j].imshow(target_img)
                ax[1, j].set_title("Target")

                ax[2, j].imshow(output_img)
                ax[2, j].set_title("Prediction")

            # if running in a notebook, display the plot
        if is_notebook():
            plt.show()
        else:
            plt.savefig(os.path.join(
                config.results_dir, f"results_{epoch}.png"))

        model.train()


def plot_loss(losses: Sequence[float]):
    '''
    Plot the loss over time
    '''
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    if is_notebook():
        plt.show()
    else:
        plt.savefig(os.path.join(config.results_dir, "loss.png"))


def split_loaders(dataset: Dataset, train_size=0.8):
    '''
    Split the dataset into training and validation loaders
    '''
    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader
