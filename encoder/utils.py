import torch
from torch import nn
from typing import Tuple
from base.dataset import GenericDataset

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, name: str) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, name)
    
def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str) -> Tuple[int, nn.Module, torch.optim.Optimizer]:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], model, optimizer

def save_data(dataset: GenericDataset, name: str) -> None:
    ds_obj = {
        'inputs': dataset.input_loader.get_ids(),
        'targets': dataset.target_loader.get_ids()
    }

    torch.save(ds_obj, name)