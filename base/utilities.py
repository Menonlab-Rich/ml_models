from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from typing import Sequence


class Loss(ABC):
    def __init__(cls) -> None:
        cls.losses = []
    
    @abstractmethod
    def __call__(cls, y_pred: Tensor, y_true: Tensor) -> Tensor:
        pass
    
    @abstractmethod
    def plot(cls, epochs: Sequence[int]):
        pass
    
    

    

class BaseUtilities(ABC):
    '''
    Abstract class for a model utilities
    '''
    
    @classmethod
    @abstractmethod
    def save_checkpoint(cls, model: nn.Module, optimizer: nn.Module, epoch: int, path: str):
        '''
        Save the model
        '''
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, path)
         
    @classmethod
    @abstractmethod
    def load_checkpoint(cls, model: nn.Module, optimizer: nn.Module, path: str):
        '''
        Load the model
        '''
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return model, optimizer, epoch
    
    @classmethod
    @abstractmethod
    def save_data(cls, dataset: Dataset, name: str):
        '''
        Save the data
        '''
        pass
    
    @classmethod
    @abstractmethod
    def predict(cls, model: nn.Module, x: Tensor):
        '''
        Make a prediction
        '''
        pass
    
    @classmethod
    @abstractmethod
    def evaluate(cls, model: nn.Module, x: Tensor, y: Tensor):
        '''
        Evaluate the model
        '''
        pass
    
    @classmethod
    @abstractmethod
    def save_example(cls, model: nn.Module, x: Tensor, y: Tensor, epoch: int, path: str):
        '''
        Save an example
        '''
        pass