import torchvision.models as models
from torch import nn
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix
from neptune.types import File
import warnings


class ResNet(pl.LightningModule):
    """
    ResNet model for PyTorch Lightning.

    This class defines a ResNet-based neural network model for binary classification, 
    incorporating an optional encoder for feature extraction.
    """

    def __init__(self, n_classes, n_channels=3, encoder=None, lr=1e-3, **kwargs):
        """
        Initialize the ResNet model.

        Parameters:
        n_classes (int): Number of output classes.
        n_channels (int): Number of input channels (default: 3).
        encoder (nn.Module): Optional encoder model for feature extraction.
        lr (float): Learning rate (default: 1e-3).
        kwargs (dict): Additional hyperparameters.
        """
        super().__init__()
        default_hparams = {
            'n_classes': n_classes,
            'n_channels': n_channels,
            'lr': lr,
        }
        hparams = {**default_hparams, **kwargs}
        # Remove any callable hyperparameters
        hparams = {k: v for k, v in hparams.items() if not callable(v)}
        hparams['encoder'] = encoder.__class__.__name__ if encoder else None
        self.save_hyperparameters(hparams)
        self.encoder = encoder.to(self.device) if encoder else None

        # Define metrics for validation and testing
        self.validation_accuracy = BinaryAccuracy()
        self.validation_bcm = BinaryConfusionMatrix(normalize='true')
        self.test_accuracy = BinaryAccuracy()
        self.test_bcm = BinaryConfusionMatrix(normalize='true')

        # Initialize ResNet backbone
        backbone = models.resnet50(weights="DEFAULT")
        n_filters = backbone.fc.in_features

        # Adjust the first layer to accept n_channels if it is not 3
        if n_channels != 3:
            backbone.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # Define classifier and loss function
        self.classifier = nn.Linear(n_filters, n_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

        if encoder:
            self.encoder.to(self.device)  # Move encoder to device
            self.encoder.eval()  # Freeze the encoder

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output predictions.
        """
        with torch.no_grad():
            if self.encoder:
                x, _ = self.encoder(x)  # Get embeddings from encoder
        features = self.feature_extractor(x)
        preds = self.classifier(features.squeeze())
        return preds.squeeze()

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
        torch.optim.Optimizer: Configured optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _shared_step(self, batch, batch_idx):
        """
        Common code shared across different step methods (training, validation, testing).

        Parameters:
        batch (tuple): A batch of data.
        batch_idx (int): Batch index.

        Returns:
        tuple: Input tensor, ground truth labels, and predictions.
        """
        x, y, _ = batch
        y_hat = self(x)
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)  # Add batch dimension if needed
        return x, y, y_hat

    def _log_metrics(self, log_location, **metrics):
        """
        Logs metrics to the specified location.

        Parameters:
        log_location (str): The base location for logging metrics.
        metrics (dict): Key-value pairs of metric names and their values.
                        If the value is a dictionary with a 'metric_value' key, the dictionary 
                        Is treated as a special case with optional 'metric_name', 'cast', and 'pickle' keys.
        """
        for metric, value in metrics.items():
            # Check if the value is a tuple with a type conversion function
            pickle = None  # Default behavior
            if type(value) == dict and 'metric_value' in value:
                metric = value['metric_name'] if 'metric_name' in value else metric
                value = value['metric_value']
                if 'cast' in value:
                    value = value['cast'](value)
                if 'pickle' in value:
                    pickle = value['pickle']
                continue
            try:
                # the pickle argument is used to force pickling or not pickling
                if pickle == True:
                    self.logger.experiment[log_location] = File.as_pickle(value)
                    continue
                if pickle == False and pickle is not None:
                    can_log = True
                    log_type = type(value)
                else:
                    can_log, log_type = self._is_loggable_without_pickle(value)

                _log_location = f"{log_location}/{metric}"

                if can_log:
                    val = log_type(value)
                    self.log(_log_location, val, on_step=False,
                             on_epoch=True, prog_bar=True, logger=True)
                else:
                    if self._is_figure(value):
                        self.logger.experiment[_log_location] = File.as_image(
                            value)
                    else:
                        self.logger.experiment[_log_location] = File.as_pickle(
                            value)
            except Exception as e:
                warnings.warn(
                    f"Could not log metric '{metric}' due to error: {e}")

    def _is_figure(self, figure):
        """
        Determines if the given object is a matplotlib Figure.

        Parameters:
        figure (object): The object to check.

        Returns:
        bool: True if the object is a Figure, False otherwise.
        """
        return figure.__class__.__name__ == 'Figure'

    def _is_loggable_without_pickle(self, obj):
        """
        Checks if an object can be logged without pickling.

        Parameters:
        obj (object): The object to check.

        Returns:
        tuple: (bool, type) indicating if the object can be logged without pickling and its loggable type.
        """
        from numpy import ndarray as np_ndarray
        from matplotlib.figure import Figure
        from torch import Tensor

        # Torch tensors and figures are common enough that they merit special handling
        # All other special handling should be done through the special case dictionary

        if self._is_figure(obj):
            return False, None  # Figures are not loggable without pickling

        if isinstance(obj, Tensor) and obj.dim() > 0:
            return False, None
        
        if isinstance(obj, Tensor): # Scalars are loggable
            return True, obj.item() 

        if isinstance(obj, (int, float, str, bool, type(None))):
            return True, type(obj)

        # Check if the object is a simple numpy type
        if obj.__class__.__module__ == 'numpy' and not isinstance(
                obj, np_ndarray):
            return True, type(obj)

        # Can the object be cast to a meaningful string?
        try:
            obj_str = str(obj)
            can_str = obj_str and not obj_str.startswith(
                "<") and not obj_str.endswith(">") and len(obj_str) < 1000
            if can_str:
                return True, str
        except:
            return False, None

        return False, None

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Parameters:
        batch (tuple): A batch of data.
        batch_idx (int): Batch index.

        Returns:
        torch.Tensor: Computed loss.
        """
        x, y, y_hat = self._shared_step(batch, batch_idx)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Parameters:
        batch (tuple): A batch of data.
        batch_idx (int): Batch index.

        Returns:
        torch.Tensor: Computed loss.
        """
        x, y, y_hat = self._shared_step(batch, batch_idx)
        loss = self.loss_fn(y_hat, y)
        self.validation_accuracy.update(y_hat, y)
        self.validation_bcm.update(y_hat, y)
        self.log('training/validation/loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('training/validation/accuracy_step', self.validation_accuracy.compute(),
                 on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Parameters:
        batch (tuple): A batch of data.
        batch_idx (int): Batch index.

        Returns:
        torch.Tensor: Computed accuracy.
        """
        # assert model is in eval mode (just to make sure)
        assert not self.training
        x, y, y_hat = self._shared_step(batch, batch_idx)
        loss = self.loss_fn(y_hat, y)
        self.test_accuracy.update(y_hat, y)
        self.log('testing/loss', loss, on_step=True,
                on_epoch=True, prog_bar=True, logger=True)
        self.log('testing/accuracy_step', self.test_accuracy.compute().item(),
                on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

        
        
    def on_test_epoch_end(self):
        """
        Actions to perform at the end of the test epoch.
        """
        assert not self.training
        epoch = self.current_epoch
        fig, ax = self.test_bcm.plot(labels=['605', '625'])
        kwargs = {
                f"bcm_results_{epoch}": self.test_bcm.compute(),
                f"bcm_plot_{epoch}": fig,
                "accuracy_val": self.test_accuracy.compute().item()
        }
        self._log_metrics(
            "testing", **kwargs)

        # Log the validation accuracy so it can be monitored
        self.log('test_acc', self.test_accuracy.compute(),
                on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.test_bcm.reset()
        self.test_accuracy.reset()

    def on_validation_epoch_end(self):
        """
        Actions to perform at the end of the validation epoch.
        """
        epoch = self.current_epoch
        fig, ax = self.validation_bcm.plot(labels=['605', '625'])
        kwargs = {
                f"bcm_results_{epoch}": self.validation_bcm.compute(),
                f"bcm_plot_{epoch}": fig,
                "accuracy_val": self.validation_accuracy.compute().item()
        }
        self._log_metrics(
            "training/validation", **kwargs)

        # Log the validation accuracy so it can be monitored
        self.log('accuracy_val', self.validation_accuracy.compute(),
                 on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.validation_bcm.reset()
        self.validation_accuracy.reset()


class BCEResnet(ResNet):
    """
    Binary Classification ResNet model for PyTorch Lightning.

    This class defines a ResNet-based neural network model specifically for binary classification, 
    inheriting from the ResNet class.
    """

    def __init__(self, pos_weight=None, **kwargs):
        """
        Initialize the BCEResnet model.

        Parameters:
        weight (torch.Tensor): Optional weight tensor for the loss function.
        kwargs (dict): Additional hyperparameters.
        """
        if 'n_classes' in kwargs:
            # Remove the n_classes argument because it is not needed
            del kwargs['n_classes']
        super().__init__(1, **kwargs)
        if pos_weight:
            pos_weight = torch.tensor(pos_weight).to(self.device)
            self.loss_fn = nn.BCEWithLogitsLoss(weight=pos_weight).to(self.device)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
