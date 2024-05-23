from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Callable, Literal
from tqdm import tqdm


class BaseTrainer(ABC):
    '''
    Abstract class for a trainer
    '''

    def __init__(self, default_epochs=10, **kwargs):
        for k, v in kwargs.items():
            # Set all the attributes passed in the constructor
            setattr(self, k, v)

        if getattr(self, 'epochs', None) is None:
            self.epochs = default_epochs  # Set the default number of epochs

        self.best_loss = float('inf')  # Set the best loss to infinity initially
    @abstractmethod
    def step(self, data, *args, **kwargs) -> dict:
        '''
        Perform a single training step
        '''
        raise NotImplementedError

    @abstractmethod
    def pre_train(self):
        '''
        Setup the trainer before training
        Any setup that needs to be done before training should be done here
        '''
        pass

    def post_train(self, *args, **kwargs):
        '''
        Perform any post training steps
        '''
        pass
    
    def post_step(self, *args, **kwargs):
        '''
        Perform any post step steps
        '''
        pass
    
    def post_epoch(self, *args, **kwargs):
        '''
        Perform any post epoch steps
        '''
        pass
    
    @property
    @abstractmethod
    def training_data(self):
        '''
        Get the training data
        '''
        return getattr(self, '_training_data', None)

    def train(self, *args, **kwargs):
        '''
        Perform the training
        '''
        self.pre_train()  #Perform any pre-training steps
        tq = tqdm(range(self.epochs), desc='Epochs', position=0, leave=True)
        res = None
        for _ in tq:
            _res = self.train_step()  # Perform a single training step
            res = self.post_epoch(tq=tq, res=_res)  # Perform any post epoch steps
        self.post_train(res=res)  # Perform any post training steps

    def train_step(self, *args, **kwargs) -> dict:
        '''
        Perform a single training step
        '''
        tq = tqdm(self.training_data, desc='Training', position=1, leave=False)
        res = None
        for data in tq:
            res = self.step(data)
            self.post_step(res=res, tq=tq)
        return res
        

    @abstractmethod
    def evaluate(self):
        '''
        Evaluate the model
        '''
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        '''
        Plot the evaluation metrics
        '''
        raise NotImplementedError
    