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
import json

global collector
collector = utils.Losses()


def train(loader, model, opt, loss_fn, scaler, scheduler):
    global collector
    loop = tqdm(loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        if len(y.shape) < 4:
            y = y.unsqueeze(1)  # add channel dimension

        if len(x.shape) < 4:
            x = x.unsqueeze(1)  # add channel dimension

        opt.zero_grad()
        with (torch.cuda.amp.autocast()):
            preds = model(x)
            loss = loss_fn(preds, y)
        unscaled_loss = loss.item()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0)  # Clip gradients
        scaler.step(opt)
        scaler.update()
        loop.set_postfix(loss=unscaled_loss)
        collector += unscaled_loss
        # Adjust learning rate based on loss value using ReduceLROnPlateau
        scheduler.step(loss.item())


def main(predict_only=False):
    import os
    model = UNet(in_channels=config.CHANNELS_INPUT,
                 out_channels=config.CHANNELS_OUTPUT).to(config.DEVICE)

    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if config.LOAD_DST:
        training_set, validation_set = utils.load_datasets(
            os.path.join(config.DST_LOAD_DIR, 'dst.json'),
            config.INPUT_DIR, config.TARGET_DIR, tmp=config.TMP_DIR,
            transforms=(config.transform_input, config.transform_target,
                        config.transform_both),
            channels=(config.CHANNELS_INPUT, config.CHANNELS_OUTPUT),
            to_float=config.DATASET_TO_FLOAT,
            match="605-*")
    else:
        ds = Dataset(config.TRAIN_IMG_PATTERN, config.TARGET_IMG_PATTERN,
                     transforms=(config.transform_input, config.transform_target,
                                 config.transform_both),
                     channels=(config.CHANNELS_INPUT, config.CHANNELS_OUTPUT),
                     to_float=config.DATASET_TO_FLOAT
                     )
        training_set, validation_set = utils.split_dataset(ds)
        if config.SAVE_DST:
            parent_set = training_set.dataset
            train_inputs_indices = training_set.indices
            val_inputs_indices = validation_set.indices
            train_inputs = [parent_set.images[i] for i in train_inputs_indices]
            train_targets = [parent_set.targets[i]
                             for i in train_inputs_indices]
            val_inputs = [parent_set.images[i] for i in val_inputs_indices]
            val_targets = [parent_set.targets[i] for i in val_inputs_indices]
            datasets = {"train_inputs": train_inputs,
                        "train_targets": train_targets,
                        "val_inputs": val_inputs, "val_targets": val_targets}
            with open(os.path.join(config.DST_SAVE_DIR, "train.json"), "w") as f:
                json.dump(datasets, f)
    # Set the input and target readers if required
    # This has to happen after the dataset is created
    if config.INPUT_READER:
        if config.LOAD_DST:
            training_set.input_reader = config.INPUT_READER
            validation_set.input_reader = config.INPUT_READER
        else:
            ds.set_args(input_reader=config.INPUT_READER,)
    if config.TARGET_READER:
        if config.LOAD_DST:
            training_set.target_reader = config.TARGET_READER
            validation_set.target_reader = config.TARGET_READER
        else:
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.2, patience=2, min_lr=0.001)
    if predict_only:
        utils.save_examples(
            model, val_loader, 0, config.EXAMPLES_DIR, config.DEVICE,
            save_filenames=True)
        return
    for epoch in range(config.NUM_EPOCHS):
        train(train_loader, model, opt, config.LOSS_FN, scaler, scheduler)
        try:
            utils.save_examples(model, val_loader, epoch,
                                config.EXAMPLES_DIR, config.DEVICE)
            utils.gen_evaluation_report(
                model, val_loader, config.DEVICE, config.TASK,
                multi_channel=True if config.CHANNELS_OUTPUT > 1 else False)
        except Exception as e:
            logging.error(f"Could not save examples: {e}")
            # continue training even if examples could not be saved
        if config.SAVE_MODEL:
            utils.save_checkpoint(model, opt, filename=config.CHECKPOINT)
    try:
        global collector
        collector.plot(os.path.join(config.EXAMPLES_DIR, "/losses.png"))
    except Exception as e:
        logging.error(f"Could not save losses: {e}")


if __name__ == "__main__":
    main(predict_only=True)
