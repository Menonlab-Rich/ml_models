from base import train
from config import config
from torch import nn, optim
from warnings import warn
from base.dataset import GenericDataset
from typing import Literal
from utils import utils, Evaluator
from os import path
from torch.utils.data import DataLoader as Dataloader
import logging
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class Trainer(train.BaseTrainer):
    def __init__(self, model: nn.Module, dataset: GenericDataset, config: dict):
        device: Literal['cuda', 'cpu'] = config['device']
        if config['training']['optimizer'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['training']['lr'],
                **config['training']['optimizer_params'])
        elif config['training']['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=config['training']['lr'],
                **config['training']['optimizer_params'])
        else:
            warn('Optimizer not recognized, passed as is.')
            optimizer = config['training']['optimizer']
        self.dataset = dataset
        self.config = config
        self.model = model
        self.loss_fn = config['loss_fn']
        self.device = device
        self.optimizer = optimizer
        self.evaluator = None  # Placeholder for the evaluator
        self.logger = logging.getLogger(__name__)
        self.scaler = GradScaler()

    def pre_train(self):
        if self.config['training']['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **self.config['training']['scheduler_params'])
        elif self.config['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **self.config['training']['scheduler_params'])
        else:
            warn('Scheduler not recognized, passed as is.')
            self.scheduler = self.config['training']['scheduler']

        self.train_set, self.val_set = self.dataset.split(
            config['training']['train_ratio'])
        utils.save_data(
            self.train_set, path.join(
                self.config['directories']['data'],
                'train_set'))
        utils.save_data(
            self.val_set, path.join(
                self.config['directories']['data'],
                'val_set'))
        self.train_loader = Dataloader(
            self.train_set, batch_size=self.config['training']['batch_size'],
            shuffle=True)
        self.val_loader = Dataloader(self.val_set,
                                     batch_size=self.config['training']
                                     ['batch_size'],
                                     shuffle=False)
        self.evaluator = Evaluator(
            self.model, self.val_loader, self.loss_fn, self.device, self.config,
            [0, 1, 1])

    def train(self):
        self.pre_train()  # Setup the training process
        best_loss = float('inf')
        for epoch in range(self.config['training']['epochs']):
            self.logger.info(
                f'Epoch {epoch + 1}/{self.config["training"]["epochs"]}')
            self.train_epoch()
            val_loss = self.evaluate()
            self.scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                utils.save_checkpoint(
                    self.model, self.optimizer, epoch, self.config
                    ['directories']['model'])
                self.logger.info('Model saved')

        self.plot()

    def train_epoch(self):
        self.model.train()
        self.model.to(self.device)
        training_loop = tqdm(self.train_loader, total=len(self.train_loader))
        for inputs, targets in training_loop:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
            accuracy = self.evaluator.accuracy(outputs, targets)
            self.evaluator.update(loss, accuracy, config['training']['batch_size'])
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            training_loop.set_postfix({'loss': loss.item(), 'accuracy': accuracy.item()})
    
    def post_train(self, *args, **kwargs):
        self.scheduler.step()
    def evaluate(self):
        self.evaluator.evaluate(self.val_loader)  # Evaluate the model
        return self.evaluator.losses_per_epoch[-1] # Return the loss

    def plot(self):
        self.evaluator.plot(metrics='both', output_path=path.join(
            self.config['directories']['predictions'], 'metrics.png'))


if __name__ == '__main__':
    from model import UNet
    from base.dataset import GenericDataset as Dataset
    from config import config
    model = UNet(
        config['model']['in_channels'],
        config['model']['out_channels'],
        config['model']['features']).to(
        config['device'])
    dataset = Dataset(
        config['input_loader'],
        config['target_loader'],
        config['transform'])

    trainer = Trainer(model, dataset, config)
    trainer.train()
