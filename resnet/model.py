import torchvision.models as models
from torch import nn
import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix
from neptune.types import File


class SqueezeLaer(nn.Module):
    '''
    A custom layer to remove the extra dimension
    In cases where the channel dimension is 1, this layer will remove it
    It will do nothing if the channel dimension is greater than 1
    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x.squeeze()


def get_model(n_classes, freeze_layers=False, n_channels=3):
    resnet = models.resnet50(weights="DEFAULT")
    if freeze_layers:
        for param in resnet.parameters():
            param.requires_grad = False

    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, n_classes)

    if n_channels != 3:
        # Change the first layer to accept n_channels
        resnet.conv1 = nn.Conv2d(
            n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Squeeze the output to remove the extra dimension
    resnet = nn.Sequential(resnet, SqueezeLaer())

    # Use DataParallel to split the model across multiple GPUs
    return resnet


class ResNet(pl.LightningModule):
    def __init__(self, n_classes, n_channels=3, encoder=None, lr=1e-3, **kwargs):
        super().__init__()
        default_hparams = {
            'n_classes': n_classes,
            'n_channels': n_channels,
            'lr': lr,
        }
        hparams = {**default_hparams, **kwargs}
        # remove any callable hyperparameters
        hparams = {k: v for k, v in hparams.items() if not callable(v)}
        hparams['encoder'] = encoder.__class__.__name__ if encoder else None
        self.save_hyperparameters(hparams)
        self.encoder = encoder.to(self.device) if encoder else None
        self.accuracy = BinaryAccuracy() # Accuracy metric
        self.bcm = BinaryConfusionMatrix(normalize='true') # Confusion matrix metric
        backbone = models.resnet50(weights="DEFAULT")
        n_filters = backbone.fc.in_features
        if n_channels != 3:
            # Change the first layer to accept n_channels
            backbone.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = list(backbone.children())[:-1]
        # Squeeze the output to remove the extra dimension
        self.feature_extractor = nn.Sequential(*layers)

        # Freeze the feature extractor except in the case that the first layer is changed
        # If the first layer is changed, the feature extractor will be frozen from the second layer
        starting_layer = 1 if n_channels != 3 else 0
        for layer in self.feature_extractor[starting_layer:]:
            for param in layer.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(n_filters, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.encoder = encoder
        if encoder:
            self.encoder.to(self.device)  # Move to the device
            self.encoder.eval()  # Freeze the encoder

    def forward(self, x):
        with torch.no_grad():
            if self.encoder:
                x, _ = self.encoder(x) # Get the embeddings and ignore the decoder output
            features = self.feature_extractor(x)
        preds = self.classifier(features.squeeze())
        return preds.squeeze() # Remove the extra dimension

    def _step(self, batch, batch_idx, log_metrics=['loss', 'acc']):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        metric_w_prefixes = [metric.split('_') for metric in log_metrics]
        metrics = [mp[0] if len(mp) == 1 else mp[1] for mp in metric_w_prefixes]
        prefixes = [mp[0] if len(mp) == 2 else '' for mp in metric_w_prefixes]
        if 'loss' in metrics:
            for prefix in prefixes:
                self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if 'acc' in metrics:
            for prefix in prefixes:
                self.log(f'{prefix}_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.bcm.update(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, log_metrics=['val_loss', 'val_acc'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def test_step(self, batch, batch_idx):
        # Check the accuracy of the model
        x, y, _ = batch
        y_hat = self(x)
        # Check the accuracy
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0) # Add the batch dimension
        self.accuracy.update(y_hat, y)
        self.bcm.update(y_hat, y)
        return self.accuracy.compute()

    def on_test_epoch_end(self):
        self.log('test_acc', self.accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        fig, ax = self.bcm.plot()
        self.logger.experiment["test_bcm_plot"] = File.as_image(fig)
    
    def on_train_epoch_end(self) -> None:
        fig, ax = self.bcm.plot(labels=['605', '625']) # TODO: Make the labels dynamic
        self.logger.experiment["train/epoch_bcm_plot"] = File.as_image(fig) # Save the plot
        self.logger.experiment["train/epoch_bcm_results"] = File.as_pickle(self.bcm.compute()) # Save the results


class BCEResnet(ResNet):
    def __init__(self, weight=None, **kwargs):
        super().__init__(1, **kwargs)  # 1 classes for binary classification
        if weight:
            weight = torch.tensor(weight).to(self.device)
            self.loss_fn = nn.BCEWithLogitsLoss(weight=weight).to(self.device)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
