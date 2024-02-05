from typing import Any
import config
import torch
import logging
import matplotlib.pyplot as plt


def save_examples(model, val_loader, epoch, folder, device):
    if not hasattr(save_examples, "fixed_samples"):
        accumulated_x, accumulated_y = [], []
        for batch in val_loader:
            batch_x, batch_y = batch[0].to(device), batch[1].to(device)
            accumulated_x.append(batch_x)
            accumulated_y.append(batch_y)
            if sum([x.shape[0] for x in accumulated_x]) >= 6:
                break
        x = torch.cat(accumulated_x, dim=0)[:6]
        y = torch.cat(accumulated_y, dim=0)[:6]
        save_examples.fixed_samples = (x, y)
    else:
        x, y = save_examples.fixed_samples

    model.eval()
    with torch.no_grad():
        y_fake = model(x)
        # Normalize y_fake from [0, 255] to [0, 1] for matplotlib
        y_fake = y_fake / 255.0
    
    # Assuming x is already normalized to [0, 1]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(6):
        row = i // 3
        col = i % 3
        # Display input (grayscale) image
        axs[row, col].imshow(x[i].cpu().squeeze().numpy(), cmap='gray', interpolation='nearest')
        axs[row, col].set_title(f"Input {i+1}")
        axs[row, col].axis('off')
        
        # Display predicted (RGB) image
        # Ensure y_fake is permuted from [C, H, W] to [H, W, C] for correct display
        axs[(row+1)%2, col].imshow(y_fake[i].cpu().detach().numpy().transpose(1, 2, 0))
        axs[(row+1)%2, col].set_title(f"Prediction {i+1}")
        axs[(row+1)%2, col].axis('off')

    plt.tight_layout()
    plt.savefig(f"{folder}/comparison_epoch_{epoch}.png")
    plt.close('all')
    
    model.train()


def save_checkpoint(model, optimizer, filename):
    '''
    Save the model and optimizer state to a file
    
    Parameters:
    ----------
    model: torch.nn.Module
        Model to save
    optimizer: torch.optim.Optimizer
        Optimizer to save
    filename: str
        Filename to save the model and optimizer state to
    '''
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def check_accuracy(loader, model, device="cuda", return_outputs=0):
    '''
    Compare the accuracy of the model using
    L2 loss and Correlation coefficient

    Parameters:
    ----------
    loader: torch.utils.data.DataLoader
        DataLoader containing the data to check accuracy on
    model: torch.nn.Module
        Model to check accuracy of
    device: str, Default: "cuda"
        Device to use for the model
    return_outputs: int, Default: 0
        Number of outputs to return

    Returns:
    --------
    None if return_outputs is 0 else a list of tuples containing
    (input, target, output)
    '''
    import numpy as np
    num_correct = 0
    num_pixels = 0
    coeffs = []
    losses = []
    outputs = []

    def should_append_outputs(p=0.5):
        '''
        Randomly decide whether to append outputs to the list
        Given that the number of outputs is less than return_outputs
        and a random number is less than probability p

        Parameters:
        ----------
        p: float, Default: 0.5
            Probability of returning True
        '''
        import random

        # clip p between 0 and 1
        p = min(1, max(0, p))
        return return_outputs > 0 and len(outputs) < return_outputs and p > random.random()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        if should_append_outputs():
            outputs.append((x, y, output))
        # compare y and output
        # L2 loss
        l2_loss = torch.nn.MSELoss()(output, y)
        # Correlation coefficient
        # convert to numpy
        output = output.cpu().numpy()
        coeff = np.corrcoef(output, y)
        if coeff >= 0.9:
            num_correct += 1
        coeffs.append(coeff)
        losses.append(l2_loss)

    print(f"Got {num_correct} / {len(loader)} correct with avg coeff {np.mean(coeffs)} and loss {np.mean(losses)}")


def split_dataset(dataset, split=0.8):
    '''
    Split the dataset into train and validation sets

    Parameters:
    ----------
    dataset: torch.utils.data.Dataset
        Dataset to split
    split: float, Default: 0.8
        Fraction of the dataset to use for training

    Returns:
    --------
    datasets: List[torch.utils.data.Dataset]
        The train and validation datasets in that order
    '''

    assert split > 0 and split < 1, "split must be in the range (0, 1)"

    return torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * split), len(dataset) - int(len(dataset) * split)]
    )


class LoggerOrDefault():
    '''
    LoggerOrDefault is a class that provides a logger object
    that can be used to log messages. If no logger is provided,
    a default logger is created and used.
    '''
    _logger = None

    def __init__(self) -> None:
        pass

    @classmethod
    def logger(cls, logger=None):
        '''
        Returns a logger object. If no logger is provided, a default
        logger is created and returned. Otherwise, the provided logger
        is returned. On the first call, the logger is created and stored
        in a class variable. Subsequent calls return the same logger.

        Parameters:
        ----------
        logger: logging.Logger, Default: None
            Logger to return. If None, a default logger is created and returned
            Or, if a logger was previously provided, the same logger is returned
        '''
        if logger is not None:
            cls._logger = logger
        if cls._logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            cls._logger = logger

        return cls._logger
