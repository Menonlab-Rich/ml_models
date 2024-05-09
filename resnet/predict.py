from model import get_model
import utils
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import GenericDataset, GenericDataLoader
import config
from os import path
from PIL import Image
import numpy as np



class DatasetLoader(GenericDataLoader):
    def __init__(self, dataset_path, root_path):
        self.inputs, self.targets = utils.load_data(dataset_path)
        self.root_path = root_path
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self._read(self.inputs[idx]), self.targets[idx]
    
    def __iter__(self):
        for i, t in zip(self.inputs, self.targets):
            yield self._read(i), t
    
    def _read(self, file):
        return np.array(Image.open(path.join(self.root_path, file)))
    
class InputLoader(GenericDataLoader):
    def __init__(self, DatasetLoader):
        self.loader = DatasetLoader
    
    def __len__(self):
        return len(self.loader)
    
    def __getitem__(self, idx):
        input, _ = self.loader[idx]
        return input
    
    def __iter__(self):
        for i in self.loader:
            yield i[0]

class TargetLoader(GenericDataLoader):
    def __init__(self, DatasetLoader):
        self.loader = DatasetLoader
    
    def __len__(self):
        return len(self.loader)
    
    def __getitem__(self, idx):
        _, target = self.loader[idx]
        return target
    
    def __iter__(self):
        for i in self.loader:
            yield i[1]
        

def predict():
    
    model = get_model(config.NUM_CLASSES, n_channels=config.NUM_CHANNELS)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    utils.load_checkpoint(model, optimizer, 'cpu', config.MODEL_PATH)
    dataset = GenericDataset(config.INPUT_LOADER, config.TARGET_LOADER, config.TRANSFORMS)
    dataset.eval()
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    evaluator = utils.Evaluator(model, loader, config.LOSS_FN, config.DEVICE, x_axis='file', report=True, report_path=config.REPORT_PATH)
    evaluator.evaluate()
    evaluator.plot(metrics='both', output_path=config.PREDICTIONS_PATH)
    
    
if __name__ == '__main__':
    predict()