'''
An implementation of a simple autoencoder model using PyTorch.
Author: Rich Baird (2024)
'''

import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchmetrics import Metric
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM, PeakSignalNoiseRatio as PSNR
from typing import Union, List, Type, Callable

class MeanSquaredError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input, target)

class WeightedMSEMetric(MeanSquaredError):
    def __init__(self, weights=None, scale=1.0, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float)
            self.register_buffer('weight', weights)
        else:
            self.weight = None
        self.register_buffer('scale', torch.tensor(scale))

    def forward(self, input: torch.Tensor, target: torch.Tensor, classes=None):
        assert input.device == target.device, 'Input and target must be on the same device'

        if self.weight is not None:
            assert self.weight.device == input.device, 'Weights must be on the same device as input'
            if classes is not None:
                device = self.weight.device
                weights = torch.tensor(
                    [self.weight[cls] for cls in classes],
                    dtype=torch.float).to(device)
            normalized_weight = weights / weights.sum()
            normalized_weight = normalized_weight.view(
                -1, 1, 1, 1).expand_as(input)
            loss = (normalized_weight * (input - target) ** 2).mean()
        else:
            loss = ((input - target) ** 2).mean()

        loss = loss * self.scale

        return loss



class LitAutoencoder(pl.LightningModule):
    def __init__(
            self, input_channels: int, embedding_dim: int,
            rescale_factor=1.0, size=None, **hyper_params) -> None:
        super().__init__()

        self.loss_fn = MeanSquaredError().to(self.device) # Default loss function
        
        default_hyper_params = {
            'lr': 1e-3,
            'input_channels': input_channels,
            'embedding_dim': embedding_dim,
            'rescale_factor': rescale_factor,
            'size': size
        }

        hparams = {**default_hyper_params, **hyper_params}
        hparams = {k: v for k, v in hparams.items() if not callable(v)}

        # Save hyperparameters
        self.save_hyperparameters(hparams, ignore='loss_fn')
        self.register_buffer(
            'rescale_factor', torch.tensor(
                rescale_factor, dtype=torch.float32))
        self.register_buffer(
            'size', torch.tensor(size, dtype=torch.long)
            if size is not None else None)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                embedding_dim, 64, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, input_channels, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        mode = 'bilinear' if (
            self.size is not None and self.size[-1] > x.shape[-1]) or self.rescale_factor > 1.0 else 'nearest'
        if self.size is not None:
            x = F.interpolate(x, size=list(self.size), mode=mode)
        else:
            x = F.interpolate(x, scale_factor=self.rescale_factor, mode=mode)
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        if self.size is not None:
            reconstruction = F.interpolate(
                reconstruction, size=original_size, mode=mode)
        return embedding, reconstruction

    def _step(self, batch, batch_idx, log_metric: str):
        inputs, targets = batch
        embedding, reconstruction = self(inputs)
        loss = self.loss_fn(reconstruction, targets)
        self.log(log_metric, loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train_loss')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val_loss')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'test_loss')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])


class WeightedLitAutoencoder(LitAutoencoder):
    def __init__(self, class_names: List[str], weights: List[float], **kwargs):
        '''
        LightiningModule for a weighted autoencoder
        Args:
            weights: The weights to apply to the loss
            class_names: The names of the classes
        '''
        super().__init__(**kwargs)
        self.class_names = class_names
        if 'loss_scale' in kwargs:
            self.loss_fn = WeightedMSEMetric(weights=weights, scale = self.hparams['loss_scale']).to(self.device) # Custom loss function
        else:
            self.loss_fn = WeightedMSEMetric(weights=weights).to(self.device)

    def _compute_accuracy(self, input, target):
        ssim = SSIM()(input, target)
        psnr = PSNR()(input, target)
        return {'ssim': ssim, 'psnr': psnr}

    def _step(self, batch, batch_idx, log_metric: str):
        inputs, targets, rest = batch
        classes = rest[0] # classes are the last element in the batch
        classes = classes.to(self.device) # Move to the device
        _, reconstruction = self(inputs)
        loss = self.loss_fn(reconstruction, targets, classes)
        accuracy = self._compute_accuracy(reconstruction, targets)
        self.log(log_metric, loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        accuracy_metrics = accuracy.keys()
        for metric in accuracy_metrics:
            self.log(f'accuracy_{metric}', accuracy[metric], on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
        return loss


class Autoencoder(nn.Module):
    def __init__(
            self, input_channels, embedding_dim, rescale_factor=1.0, size=None):
        '''
        A simple autoencoder model
        Args:
            input_channels: The number of channels in the input image
            embedding_dim: The size of the embedding space
            rescale_factor: The factor by which to rescale the input image before encoding
                default: 1.0 (no rescaling)
        '''
        super(Autoencoder, self).__init__()
        if size is not None:
            self.register_buffer('size', torch.tensor(size, dtype=torch.long))
        self.register_buffer(
            'rescale_factor', torch.tensor(
                rescale_factor, dtype=torch.float32))
        self._encoder_model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self._decoder_model = nn.Sequential(
            nn.ConvTranspose2d(
                embedding_dim, 64, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, input_channels, kernel_size=3, stride=2, padding=1,
                output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        original_size = x.shape[-2:]
        if hasattr(self, 'size'):
            mode = 'bilinear' if self.size[-1] > x.shape[-1] else 'nearest'
            x = F.interpolate(x, size=list(self.size), mode=mode)
        else:
            mode = 'bilinear' if self.rescale_factor > 1.0 else 'nearest'
            x = F.interpolate(x, scale_factor=self.rescale_factor, mode=mode)
        embedding = self._encoder_model(x)
        reconstruction = self._decoder_model(embedding)
        if hasattr(self, 'size'):
            reconstruction = F.interpolate(
                reconstruction, size=original_size, mode=mode)
        return embedding, reconstruction

    @property
    def encoder(self):
        return self._encoder_model

    @property
    def decoder(self):
        return self._decoder_model


if __name__ == '__main__':
    model = WeightedLitAutoencoder(input_channels=1, embedding_dim=32, size=(128, 128), weights=[1.0, 1.0], class_names=['class1', 'class2'])
    print(model)
    x = torch.randn(1, 1, 128, 128)
    embedding, reconstruction = model(x)
    print(embedding.shape, reconstruction.shape)
    print('Success!')
