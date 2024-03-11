import torch
import config
from Model import MobileNetV2AF
from torch.utils.data import DataLoader

def train():
    
    current_steps = 0
    model = MobileNetV2AF(num_input_channels=1).to(config.DEVICE) 
    adam = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    #TODO: Create a dataset and a DataLoader
    dl = DataLoader(batch_size=config.BATCH_SIZE) # Add the DataLoader here
    while current_steps < config.NUM_STEPS:
        for x, y in dl:
            x, y = x.to(config.DEVICE).float(), y.to(config.DEVICE).float() # Move the inputs and targets to the device
            
            adam.zero_grad()
            with config.AUTOCAST:
                output = model(x)
                loss = config.ORDINAL_REGRESSION_LOSS(output, y)
                config.GRAD_SCALER.scale(loss).backward() # Calculate the gradients
                config.GRAD_SCALER.step(adam) # Update the weights
                config.GRAD_SCALER.update() # Update the scaler
            current_steps += 1
    