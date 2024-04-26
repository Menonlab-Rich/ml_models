from abc import abstractmethod, ABC
import torch
from torch import nn
from torch.utils.data import DataLoader

class BaseConfigHandler(ABC):
    '''
    Abstract class for a config handler
    '''
    @abstractmethod
    def __init__(self):
        self.config = {}
    
    @abstractmethod
    def save(self, path: str):
        '''
        Save the config
        '''
        pass
    
    @abstractmethod
    def load(self, path: str):
        '''
        Load the config
        '''
        pass
    
    def __getitem__(self, key):
        return self.config.get(key, None)
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def __iter__(self):
        for key, value in self.config.items():
            yield key, value

class TomlConfigHandler(BaseConfigHandler):
    from toml import load, dump
    from warnings import warn
    def __init__(self) -> None:
        self.config = {}
        self.callabes = {}
    
    def load(self, path: str, callables: dict = {}):
        '''
        Load a config file
        
        Parameters:
        path (str): The path to the file
        callables (dict): A dictionary of callables to use for the config
        '''
        try:
            with open(path, 'r') as f:
                self.config = self.load(f)
            for key, _ in self.config.items():
                if key in callables:
                    self.config[key] = callables[key]    
        except FileNotFoundError:
            print(f"File not found: {path}")
    
    def save(self, path: str):
        with open(path, 'w') as f:
            self.dump(self.config, f)
    
    def __getitem__(self, key: str) -> any:
        key_value = self.config.get(key, None)
        if key_value in self.callables:
            return self.callables[key_value]
        
        return key_value
    
    def __setitem__(self, key: str, value: any) -> None:
        if callable(value):
            _id = id(value)
            self.config[key] = _id
            self.callables[_id] = value
        else:
            self.config[key] = value





