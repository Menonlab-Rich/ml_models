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

collector = utils.Losses()

def train(loader, model, opt, loss_fn, scaler):
    loop = tqdm(loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        if len(y.shape) < 4:
            y = y.unsqueeze(1)  # add channel dimension

        if len(x.shape) < 4:
            x = x.unsqueeze(1)  # add channel dimension

        with (torch.cuda.amp.autocast()):
            preds = model(x)
            loss = loss_fn(preds, y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loop.set_postfix(loss=loss.item())
        collector += loss.item()


def main(predict_only=False):
    model = UNet(in_channels=config.CHANNELS_INPUT,
                 out_channels=config.CHANNELS_OUTPUT).to(config.DEVICE)

    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    if config.LOAD_DST:
        training_set = torch.load(config.DST_SAVE_DIR + "/train.pth")
        validation_set = torch.load(config.DST_SAVE_DIR + "/val.pth")
    else:
        ds = Dataset(config.TRAIN_IMG_PATTERN, config.TARGET_IMG_PATTERN,
                    transforms=(config.transform_input, config.transform_target,
                                config.transform_both),
                    channels=(config.CHANNELS_INPUT, config.CHANNELS_OUTPUT),
                    to_float=config.DATASET_TO_FLOAT
            )
        training_set, validation_set = utils.split_dataset(ds)
    
    # Set the input and target readers if required
    # This has to happen after the dataset is created
    if config.INPUT_READER:
        ds.set_args(input_reader=config.INPUT_READER,)
    if config.TARGET_READER:
        ds.set_args(target_reader=config.TARGET_READER)
    
    
    train_loader = DataLoader(training_set, shuffle=True)
    val_loader = DataLoader(validation_set, shuffle=False)

    # Load model if required
    if config.LOAD_MODEL:
        try:
            utils.load_checkpoint(model, opt, config.CHECKPOINT)
        except Exception as e:
            logging.error(f"Could not load model: {e}")
            logging.warning("Training from scratch")

    scaler = torch.cuda.amp.GradScaler()
    if predict_only:
        utils.save_examples(model, val_loader, 0,
                            config.EXAMPLES_DIR, config.DEVICE)
        return
    for epoch in range(config.NUM_EPOCHS):
        train(train_loader, model, opt, config.LOSS_FN, scaler)
        try:
            utils.save_examples(model, val_loader, epoch,
                                config.EXAMPLES_DIR, config.DEVICE)
            utils.gen_evaluation_report(
                model, val_loader, config.DEVICE, config.TASK, multi_channel=True
                if config.CHANNELS_OUTPUT > 1 else False)
        except Exception as e:
            logging.error(f"Could not save examples: {e}")
            # continue training even if examples could not be saved
        if config.SAVE_MODEL:
            utils.save_checkpoint(model, opt, filename=config.CHECKPOINT)
    
    collector.plot(config.RESULTS_DIR + "/losses.png")


if __name__ == "__main__":
    main(predict_only=False)
