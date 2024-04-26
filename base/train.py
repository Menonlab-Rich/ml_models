from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Callable, Literal

class BaseTrainer(ABC):
    '''
    Abstract class for a trainer
    '''
    def __init__(self, model: nn.Module, dataset: Dataset, loss_fn: Callable[[torch.Tensor], float], optimizer: nn.Module, device: Literal['cuda', 'cpu'], **kwargs):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.config = kwargs
        
    @abstractmethod
    def setup(self):
        '''
        Setup the trainer before training
        Any setup that needs to be done before training should be done here
        '''
        pass
    
    @abstractmethod
    def train(self):
        '''
        Perform the training
        '''
        pass

    @abstractmethod
    def evaluate(self):
        '''
        Evaluate the model
        '''
        pass

    @abstractmethod
    def plot(self):
        '''
        Plot the evaluation metrics
        '''
        pass
    