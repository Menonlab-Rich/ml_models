import torch
import utils
from base.train import BaseTrainer
from base.dataset import GenericDataset
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from typing import Callable, Literal
from tqdm import tqdm


class Pix2PixTrainer(BaseTrainer):
    def __init__(
            self, generator: nn.Module, discriminator: nn.Module,
            dataset: GenericDataset, loss_fn_g: Callable[[torch.Tensor, torch.Tensor], float],
            loss_fn_d: Callable[[torch.Tensor, bool], float],
            optimizer_g: nn.Module, optimizer_d: nn.Module,
            device: Literal['cuda', 'cpu'],
            scaler: bool, **kwargs):

        scaler = device == 'cuda' and scaler

        config = kwargs.get('config', {})
        args = {
            'generator': generator,  # The generator model
            'discriminator': discriminator,  # The discriminator model
            'optimizer_g': optimizer_g,  # The optimizer for the generator
            'optimizer_d': optimizer_d,  # The optimizer for the discriminator
            'device': device,  # The device to train the model on
            'dataset': dataset,  # The dataset to train the model on
            'loss_fn_g': loss_fn_g,  # The loss function to use
            'loss_fn_d': loss_fn_d,  # The loss function to use for the discriminator
            'config': kwargs.get('config', {}),  # Any additional configuration
            'scheduler_g': None,  # The scaler for mixed precision training
            'scheduler_d': None,  # The scheduler for the discriminator, defaults to 'None
            **kwargs
        }

        epochs = kwargs.get('epochs', None)  # The number of epochs to train for
        if epochs is None:
            epochs = config.get('epochs', None)

        args['epochs'] = epochs

        super().__init__(**args)  # Initialize the base trainer with the arguments
        # The scaler for mixed precision training or None
        self.scaler = amp.GradScaler() if scaler else None
        # The losses for the generator and discriminator
        self.losses = {'g': [], 'd': []}
        self.best_loss_d = float('inf')  # The best loss for the discriminator
        self.best_loss_g = float('inf')  # The best loss for the generator

    def pre_train(self):
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.loss_fn_g.to(self.device)
        self.loss_fn_d.to(self.device)
        self.train_ds, self.val_ds = self.dataset.split(0.8)
        self.train_dl = DataLoader(
            self.train_ds, batch_size=self.config.get('batch_size', 1),
            shuffle=True)
        self.val_dl = DataLoader(
            self.val_ds, batch_size=self.config.get('batch_size', 1),
            shuffle=False)

        self.optimizer_d = self.optimizer_d(self.discriminator.parameters())
        self.optimizer_g = self.optimizer_g(self.generator.parameters())
        # The scheduler for the generator
        self.scheduler = self.scheduler_g(self.optimizer_g)
        if self.config.get('load_model', False):
            utils.load_checkpoint(
                self.generator, self.optimizer_g, self.config, path.join(
                    self.config['directories']['model'], 'generator.tar')
            )
            utils.load_checkpoint(
                self.discriminator, self.optimizer_d, self.config, path.join(
                    self.config['directories']['model'], 'discriminator.tar'))

    def step(self, data, *args, **kwargs):
        if data is None:
            raise ValueError('Data must be provided to the step method')

        inputs, targets = data
        inputs, targets = inputs.to(
            self.device), targets.to(self.device)

        # Train Discriminator
        with amp.autocast():
            self.optimizer_d.zero_grad()
            output_real = self.discriminator(targets)
            output_fake = self.discriminator(self.generator(inputs).detach())
            loss_d_real = self.loss_fn_d(output_real, True)
            loss_d_fake = self.loss_fn_d(output_fake, False)
            loss_d = loss_d_real + loss_d_fake

            if self.scaler:
                self.scaler.scale(loss_d).backward()
            else:
                loss_d.backward()

        self.optimizer_d.step()

        # Train Generator
        with amp.autocast():
            self.optimizer_g.zero_grad()
            generated_images = self.generator(inputs)
            output_fake = self.discriminator(generated_images)
            adversarial_loss = self.loss_fn_d(output_fake, True)
            loss_g = self.loss_fn_g(output_fake, generated_images, targets)
            loss_g = adversarial_loss + loss_g
            if self.scaler:
                self.scaler.scale(loss_g).backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.scaler.step(self.optimizer_g)
                self.scaler.update()
            else:
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.optimizer_g.step()

        return {'loss_g': loss_g.item(), 'loss_d': loss_d.item()}

    def post_step(self, res, tq=None):
        loss_g = res['loss_g']
        loss_d = res['loss_d']
        if tq:
            tq.set_postfix(loss_g=loss_g, loss_d=loss_d)
        return res

    def post_epoch(self, tq=None, res=None):
        loss_g = res['loss_g']
        loss_d = res['loss_d']
        self.losses['g'].append(loss_g)
        self.losses['d'].append(loss_d)
        self.scheduler.step()  # Step the scheduler after each epoch
        if loss_g < self.best_loss_g:
            self.best_loss_g = loss_g
            utils.save_checkpoint({
                'generator': self.generator.state_dict(),
                'optimizer_g': self.optimizer_g.state_dict()
            }, path.join(self.config['directories']['model'], 'generator.tar'))

        if loss_d < self.best_loss_d:
            self.best_loss_d = loss_d
            utils.save_checkpoint(
                {'discriminator': self.discriminator.state_dict(),
                 'optimizer_d': self.optimizer_d.state_dict()},
                path.join(
                    self.config['directories']['model'],
                    'discriminator.tar'))

    @property
    def training_data(self):
        return self.train_dl

    def train(self):
        super().train(self.step)

    def train_step(self, *args, **kwargs) -> dict:
        return super().train_step(*args, **kwargs)

    def evaluate(self):
        self.generator.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_dl):
                masks, target_images = data
                masks, target_images = masks.to(
                    self.device), target_images.to(
                    self.device)
                output = self.generator(masks)
                # Add evaluation metrics here
        self.generator.train()

    def plot(self):
        from matplotlib import pyplot as plt
        plt.plot(self.losses['g'], label='Generator Loss')
        plt.plot(self.losses['d'], label='Discriminator Loss')
        plt.legend()
        plt.savefig('losses.png')

    def post_train(self, *args, **kwargs):
        return super().post_train(*args, **kwargs)


if __name__ == '__main__':
    # First, load the dataset
    from dataset import get_dataset
    from config import Config, DiscriminatorLoss, GeneratorLoss
    from model import Generator, Discriminator
    from os import path
    conf_file = path.join(path.dirname(__file__), 'config.yml')
    config = Config(config_file=conf_file)
    gen = Generator(
        config['model']['in_channels'],
        config['model']['out_channels'])
    disc = Discriminator(
        config['model']['in_channels'],
        config['model']['out_channels'])
    dataset = get_dataset(
        config['directories']['inputs'],
        config['directories']['targets'],
        config['transform'])
    trainer = Pix2PixTrainer(gen, disc, dataset,
                             GeneratorLoss(config['model']['lambda']),
                             DiscriminatorLoss(),
                             config['optimizer'],
                             config['optimizer'],
                             config['device'],
                             scaler=True, config=config,
                             scheduler_g=config['scheduler'])

    trainer.train()
