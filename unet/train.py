import torch
import torch.nn as nn
import torch.optim as optim
import config
import utils
import logging
from dataset import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import UNet


def train(loader, model, opt, loss_fn, scaler):
    loop = tqdm(loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE).float(), y.to(config.DEVICE).float()

        if len(y.shape) < 4:
            y = y.unsqueeze(1)  # add channel dimension

        if len(x.shape) < 4:
            x = x.unsqueeze(1)  # add channel dimension

        with torch.cuda.amp.autocast():
            preds = model(x)
            loss = loss_fn(preds, y)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main():
    logger = logging.getLogger(__name__)
    model = UNet(in_channels=config.CHANNELS_INPUT,
                 out_channels=config.CHANNELS_OUTPUT).to(config.DEVICE)

    # if the output is a single channel, use BCEWithLogitsLoss to combine the sigmoid and the BCELoss
    if config.CHANNELS_OUTPUT == 1:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.L1Loss()
    
    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    ds = Dataset(config.TRAIN_IMG_DIR, config.TARGET_DIR,transform=config.training_transform)
    training_set, validation_set = utils.split_dataset(ds) # split the dataset into train and validation sets with 80% and 20% of the data respectively
    train_loader = DataLoader(training_set, shuffle=True)
    val_loader = DataLoader(validation_set, shuffle=False)
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.NUM_EPOCHS):
        train(train_loader, model, opt, loss_fn, scaler)
        utils.save_examples(model, val_loader, epoch, config.EXAMPLES_DIR)
        if config.SAVE_MODEL:
            utils.save_checkpoint(model, opt, filename=config.CHECKPOINT)

if __name__ == "__main__":
    main()
