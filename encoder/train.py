from base.train import BaseTrainer as Trainer
import torch
from model import Autoencoder
from torch.optim import Adam
from dataset import EncoderDataset, InputLoader
from os import path
import utils
from config import Config
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from base.loss import WeightedMSELoss


class EncoderTrainer(Trainer):
    def __init__(
            self, model, train_loader, val_loader, optimizer, loss_fn,
            device='cuda', epochs=10, **kwargs):
        properties = {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': optimizer,
            'loss_fn': loss_fn,
            'device': device,
            'epochs': epochs,
            'scaler': GradScaler(),  # Initialize the gradient scaler
            'losses': [],  # Initialize the losses list
            **kwargs,  # Add any additional keyword arguments
        }
        super(EncoderTrainer, self).__init__(**properties)

    def pre_step(self, *args, **kwargs):
        self.model.train()  # Set the model to training mode
        self.optimizer.zero_grad()

    def post_step(self, **kwargs):
        step_result = None
        pbar = None
        if 'res' in kwargs:
            step_result = kwargs['res']
        if 'tq' in kwargs:
            pbar = kwargs['tq']
        pbar.set_postfix({'loss': step_result['loss'].item()})

        # Perform the gradient update
        self.scaler.scale(step_result['loss']).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def step(self, data, *args, **kwargs) -> dict:
        inputs, targets, names, _ = data
        # If names has 605 and 625, then we need to split the targets and inputs
        classes = [name[:3] for name in names]
        classes = [0 if '605' in cls else 1 for cls in classes]
        inputs, targets = inputs.to(
            self.device).float(), targets.to(
            self.device).float()
        with autocast():
            encoded, decoded = self.model(inputs)
            loss = self.loss_fn(decoded, targets, classes)

        return {'loss': loss}

    def pre_train(self):
        self.model = model.to(self.device)  # Move the model to the device

    @property
    def training_data(self):
        return self.train_loader

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data in self.val_loader:
                inputs, targets = data
                encoded, decoded = self.model(inputs)
                # Calculate the loss between the decoded and the target
                loss = self.loss_fn(decoded, targets)
                total_loss += loss.item()
        return {'val_loss': total_loss / len(self.val_loader)}

    def post_epoch(self, *args, **kwargs):
        eval_res = self.evaluate()
        loss = eval_res['val_loss']
        self.losses.append(loss)
        if self.best_loss < loss:
            self.best_loss = loss
            utils.save_checkpoint(
                self.model, self.optimizer, self.epoch, path.join(
                    self.checkpoint_dir, 'auto_encoder_best.pth.tar'))
        return eval_res

    def post_train(self, *args, **kwargs):
        epoch, model, optimizer = utils.load_checkpoint(
            self.model, self.optimizer,
            path.join(self.checkpoint_dir, 'auto_encoder_best.pth.tar'))

        # freeze the encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

        # pickle and save the encoder
        torch.save(model.encoder, path.join(
            self.checkpoint_dir, 'encoder_only_best.pth.tar'))
        self.plot()

    def plot(self):
        from matplotlib import pyplot as plt
        plt.plot(self.losses)
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.savefig(path.join(self.checkpoint_dir, 'val_loss.png'))




if __name__ == '__main__':
    config = Config(file_path=r"D:\CZI_scope\code\ml_models\encoder\config.yml")
    loss_fn = WeightedMSELoss(weights=torch.tensor(
        [config.weights['605'], config.weights['625']]), scale=config.loss_scale).to(config.device)
    dataset = EncoderDataset(InputLoader(config.data_dir), config.transform)
    model = Autoencoder(
        config.input_channels, config.embedding_dim,
        rescale_factor=config.rescale)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    train_ds, val_ds = dataset.split(0.8, return_identifiers=True)
    # Save the training dataset
    utils.save_data(train_ds, path.join(config.out_dir, 'train_ds.pth.tar'))
    # Save the validation dataset
    utils.save_data(val_ds, path.join(config.out_dir, 'val_ds.pth.tar'))
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    trainer = EncoderTrainer(
        model, train_loader, val_loader, optimizer, loss_fn,
        epochs=config.epochs, device=config.device,
        checkpoint_dir=config.out_dir)

    trainer.train()
