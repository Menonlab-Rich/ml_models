import torchvision.models as models
from torch import nn
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from base.loss import with_loss_fn

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
        resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Squeeze the output to remove the extra dimension
    resnet = nn.Sequential(resnet, SqueezeLaer()) 

    # Use DataParallel to split the model across multiple GPUs
    return resnet

class ResNet(pl.LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        backbone = models.resnet50(weights="DEFAULT")
        n_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval() # Freeze the feature extractor
        self.classifier = nn.Linear(n_filters, n_classes)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return self.classifier(features.squeeze())   
    
    def _step(self, batch, batch_idx, log_metrics=['loss', 'acc']):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        if 'loss' in log_metrics:
            self.log('loss', loss)
        if 'acc' in log_metrics:
            acc = Accuracy()(y_hat, y)
            self.log('acc', acc)
        return loss

@with_loss_fn(nn.BCEWithLogitsLoss, weights=[0.25, 0.75])
class BCEResnet(ResNet):
    def __init__(self):
        super().__init__(2) # 2 classes for binary classification
        self.loss_fn = nn.BCEWithLogitsLoss()