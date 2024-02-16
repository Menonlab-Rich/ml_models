import torch
import torch.nn as nn
import torch.optim as optim
import config
from utils import save_examples, save_checkpoint, load_checkpoint
from dataset import Dataset
from discriminator import Discriminator
from generator import Generator
from tqdm import tqdm
from torch.utils.data import DataLoader


def train(
        disc, gen, train_loader, opt_disc, opt_gen, BCE, L1_LOSS, g_scaler,
        d_scaler):
    loop = tqdm(train_loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = BCE(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
            # to make the discriminator learn slower relative to the generator
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
            L1 = L1_LOSS(y_fake, y) * config.L1_LAMBDA  # L1 weight
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


def main():
    disc = Discriminator().to(config.DEVICE)
    gen = Generator().to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()  # Binary Cross Entropy
    L1_LOSS = nn.L1Loss()  # L1 loss

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen,
                        opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc,
                        opt_disc, config.LEARNING_RATE)

    dataset = Dataset(
        image_globbing_pattern=r"D:\CZI_scope\code\data\videos\training_data\stitched\*.jpg",
        target_globbing_pattern=r"D:\CZI_scope\code\data\videos\training_data\stitched\*.jpg",
        make_even=False, make_square=False, match_shape=False, target_input_combined=True, axis="x",
        transform=(None, config.transform_only_input, config.
                   transform_only_target))
    # split the dataset into train and validation sets
    train_len = int(len(dataset)*0.8)  # Use 80% of the dataset for training
    val_len = len(dataset) - train_len  # Use the remaining 20% for validation
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_len, val_len])
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train(disc, gen, train_loader, opt_disc,
              opt_gen, BCE, L1_LOSS, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_examples(
            gen, val_loader, epoch,
            folder=r"D:\CZI_scope\code\ml_models\pix2pix\eval")


def set_log_level(default_level: str):
    import logging
    import os

    log_level = os.environ.get('LOG_LEVEL', default_level)
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logging.warning(
            f'Invalid log level: {log_level}. Defaulting to {default_level}')
        numeric_level = getattr(logging, default_level.upper(), None)
    logging.basicConfig(level=numeric_level)


if __name__ == "__main__":
    set_log_level('INFO')
    main()
