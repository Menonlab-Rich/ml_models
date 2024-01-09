from typing import Any
import config
import torch
import logging
from torchvision.utils import save_image


def save_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        save_image(y_fake, f"{folder}/y_gen_{epoch}.png")
        save_image(x, f"{folder}/x_input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, f"{folder}/labelt_{epoch}.png")

    gen.train()


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, filename):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_img_dir, gt_dir, val_img_dir, val_gt_dir,
                transform_input=None, transform_target=None, batch_size=16,
                num_workers=4, pin_memory=True, gt_naming_pattern="",
                logger=None):
    from torch.utils.data import DataLoader
    from dataset import Dataset as ImageDataset
    train_ds = ImageDataset(
        train_img_dir, gt_dir, transform=transform_input,
        target_naming_pattern=gt_naming_pattern, logger=logger)
    val_ds = ImageDataset(
        val_img_dir, val_gt_dir, transform=transform_target,
        target_naming_pattern=gt_naming_pattern, logger=logger)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory)
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda", return_outputs=0):
    '''
    Compare the accuracy of the model using
    L2 loss and Correlation coefficient
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
    _logger = None

    def __init__(self) -> None:
        pass

    @classmethod
    def logger(cls, logger=None):
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
