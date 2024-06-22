import torchvision.models as models
from torch import nn
import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix
from pytorch_lightning.loggers import NeptuneLogger
from neptune.types import File
from warnings import warn


class ResNet(pl.LightningModule):
    def __init__(
            self, n_classes, n_channels=3, lr=1e-3, threshold=0.5,
            batch_size=None):
        super().__init__()
        self.save_hyperparameters()
        self.threshold = threshold
        self.validation_accuracy = BinaryAccuracy()
        self.validation_bcm = BinaryConfusionMatrix(normalize='true')
        self.test_accuracy = BinaryAccuracy()
        self.test_bcm = BinaryConfusionMatrix(normalize='true')

        backbone = models.resnet50(weights="DEFAULT")
        if n_channels != 3:
            backbone.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(backbone.fc.in_features, n_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.lr = lr

    def log(self, *args, **kwargs):
        kwargs['batch_size'] = self.hparams.get(
            'batch_size', kwargs.get('batch_size', None))
        return super().log(*args, **kwargs)

    def forward(self, x):
        features = self.feature_extractor(x).squeeze()
        return self.classifier(features)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x).view(-1)  # Flatten
        loss = self.loss_fn(y_hat, y)
        try:
            self.log('train_loss', loss, on_step=True,
                     on_epoch=True, prog_bar=True, logger=True)
        except Exception as e:
            warn(f'Error logging training metrics: {e}')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x).view(-1)  # Flatten
        loss = self.loss_fn(y_hat, y)
        self.validation_accuracy.update(y_hat, y)
        self.validation_bcm.update(y_hat, y)
        try:
            self.log('val_loss', loss, on_epoch=True, prog_bar=True)
            self.log(
                'val_accuracy', self.validation_accuracy.compute(),
                on_epoch=True, prog_bar=True)
        except Exception as e:
            warn(f'Error logging validation metrics: {e}')
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.test_accuracy.update(y_hat, y)
        try:
            self.log('test_loss', loss, on_epoch=True, prog_bar=True)
            self.log(
                'test_accuracy', self.test_accuracy.compute(),
                on_epoch=True, prog_bar=True)
        except Exception as e:
            warn(f'Error logging test metrics: {e}')
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        y_hat = self(x)
        pred = torch.sigmoid(y_hat)
        return pred > self.threshold

    def on_validation_epoch_end(self):
        try:
            fig, ax = self.validation_bcm.plot(labels=['Class 0', 'Class 1'])
            self.logger.experiment['validation_bcm_plot'].log(
                File.as_image(fig))
            self.validation_bcm.reset()
            self.validation_accuracy.reset()
        except Exception as e:
            warn(f'Error logging validation metrics: {e}')

    def on_test_epoch_end(self):
        try:
            fig, ax = self.test_bcm.plot(labels=['Class 0', 'Class 1'])
            self.logger.experiment['test_bcm_plot'].log(File.as_image(fig))
            self.test_bcm.reset()
            self.test_accuracy.reset()
        except Exception as e:
            warn(f'Error logging test metrics: {e}')


class BCEResnet(ResNet):
    def __init__(self, pos_weight=None, **kwargs):
        kwargs['n_classes'] = 1
        super().__init__(**kwargs)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight).to(self.device)
            if pos_weight else None)
