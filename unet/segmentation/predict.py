from model import UNet
from config import config
import torch
from torch.utils.data import DataLoader
from base.dataset import GenericDataset
from utils import utils, Evaluator
from os import path
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def predict():
    model = UNet(in_channels=config['model']['in_channels'], out_channels=config['model']['out_channels'], features=config['model']['features'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    utils.load_checkpoint(model, optimizer, 'cpu', config['files']['model'])
    dataset = GenericDataset(config['input_loader'], config['target_loader'], config['transform'])
    dataset.evaluate()
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
    evaluator = Evaluator(model, loader, config['loss_fn'], 'cpu', config)
    evaluator.evaluate()
    evaluator.plot(metrics='both', output_path=path.join(config['directories']['results'], 'predictions.png'))
    
def plot_examples():
    model = UNet(in_channels=config['model']['in_channels'], out_channels=config['model']['out_channels'], features=config['model']['features'])
    model.eval()
    model.to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    utils.load_checkpoint(model, optimizer, 'cpu', config['files']['model'])
    dataset = GenericDataset(config['input_loader'], config['target_loader'], config['transform'])
    dataset.evaluate()
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    colors = [(0, 0, 0), (0, 255, 0), (255, 0, 0)]
    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to('cpu')
        targets = targets.to('cpu')
        predictions = model(inputs)
        predictions = torch.argmax(predictions, dim=1)
        # plot the inputs on the first row
        plt.figure(figsize=(10, 10))
        for j in range(3):
            plt.subplot(3, 3, j + 1)
            plt.imshow(utils.prepare_img_tensors_for_plotting(inputs[j]))
            plt.title('Input')
        
        # plot the targets on the second row
        for j in range(3):
            plt.subplot(3, 3, j + 4)
            # color code the labels
            plt.imshow(utils.prepare_label_tensors_for_plotting(colors, targets[j]))
            plt.title('Target')
        
        # plot the predictions on the third row
        for j in range(3):
            plt.subplot(3, 3, j + 7)
            # color code the labels
            plt.imshow(utils.prepare_label_tensors_for_plotting(colors, predictions[j]))
            plt.title('Prediction')
        ds: GenericDataset = loader.dataset
        input_loader = ds.input_loader
        names = input_loader.get_ids(i * 3, 3)
        plt.savefig(path.join(config['directories']['results'], f'{names[0]}.png'))
        plt.close() # close the figure to avoid memory leaks
        
if __name__ == '__main__':
    # predict()
    plot_examples()