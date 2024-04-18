from Dataset import Dataset
from Model import ResNet
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import config 
import torch
import logging
import utils

def prepare(img_dir, annotations_file, with_model=True, transforms=None):
    '''
    Prepare the dataset and model
    '''
    # Create the dataset
    dataset = Dataset(image_dir=img_dir, annotations_file=annotations_file, transforms=transforms)
    # Create the model
    if with_model:
        model = ResNet.ResNet50(num_classes=config.NUM_CLASSES, img_channels=config.IMG_CHANNELS)
        model.to(config.DEVICE)
    else:
        model = None
    return dataset, model

def train(loader, model, optimizer):
    '''
    Train the model
    '''
    loop = tqdm(loader, leave=True)
    scaler = torch.cuda.amp.GradScaler() # Automatic mixed precision
    losses = []
    for _, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE).float(), y.to(config.DEVICE).float()
        with (torch.cuda.amp.autocast()):
            preds = model(x)
            loss = config.LOSS_FN(preds, y)
        optimizer.zero_grad()
        scaler.scale(loss).backward() # Backpropagation
        scaler.step(optimizer) # Update the weights
        scaler.update()
        loop.set_postfix(loss=loss.item()) # Update the progress bar
        losses.append(loss.item())
    
    return np.mean(losses) # Return the average loss

def main():
    train_dataset, model = prepare(config.TRAIN_IMG_DIR, config.TRAIN_ANNOTATIONS_FILE, transforms=config.TRANSFORMS)
    
    # Split the dataset into training and validation
    train_size = int(0.8 * len(train_dataset))
    validation_size = len(train_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, validation_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    losses = []
    # Load model if required
    if config.LOAD_MODEL:
        try:
            utils.load_checkpoint(model, optimizer, config.CHECKPOINT)
        except Exception as e:
            logging.error(f"Could not load model: {e}")
            logging.warning("Training from scratch")
    if config.PREDICT_ONLY:
        utils.save_examples(model, val_loader, 0, config.EXAMPLES_DIR, config.DEVICE)
        return
    for epoch in range(config.NUM_EPOCHS):
        loss = train(train_loader, model, optimizer)
        losses.append(loss) # Save the loss for graphing
        utils.save_examples(model, val_loader, epoch, config.EXAMPLES_DIR, config.DEVICE)
        if config.SAVE_MODEL:
            utils.save_checkpoint(model, optimizer, filename=config.CHECKPOINT)
    
    losses = np.array(losses)
    np.savez_compressed(config.LOSSES_FILE, losses=losses) # Save the losses
    
    # plot the losses
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.title('Training Loss')
    # check if we are in a Jupyter notebook
    if config.IN_JUPYTER:
        plt.show()
    else:
        plt.savefig(config.LOSS_PLOT)