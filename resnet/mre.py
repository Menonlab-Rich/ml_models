import torch
from torch import nn
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader, TensorDataset
from os import environ

class ResNet(pl.LightningModule):
    def __init__(self, n_classes=1, n_channels=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.validation_accuracy = BinaryAccuracy()
        
        backbone = models.resnet18(pretrained=True)
        n_filters = backbone.fc.in_features
        if n_channels != 3:
            backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(n_filters, n_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.squeeze())
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.validation_accuracy.update(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        val_acc = self.validation_accuracy.compute()
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.validation_accuracy.reset()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds = torch.sigmoid(y_hat) > 0.5
        return preds, y

def get_dataloader():
    # Create a simple synthetic dataset for demonstration purposes
    x = torch.randn(100, 3, 224, 224)
    y = torch.randint(0, 2, (100, 1)).float()
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8)

# Set up model, data, and trainer
model = ResNet()
train_loader = get_dataloader()
val_loader = get_dataloader()

logger = NeptuneLogger(
        api_key=environ.get("NEPTUNE_API_TOKEN"),
        project="richbai90/ResnetTest",
        tags=["MRE", "resnet"],
    )


trainer = pl.Trainer(max_epochs=3, logger=logger, log_every_n_steps=1)

# Train and validate the model
trainer.fit(model, train_loader, val_loader)

preds, targets = [], []
for batch in val_loader:
    batch_preds, batch_targets = model.predict_step(batch, 0)
    preds.extend(batch_preds)
    targets.extend(batch_targets)

# Calculate accuracy manually for comparison
preds = torch.stack(preds).view(-1)
targets = torch.stack(targets).view(-1)
manual_accuracy = (preds == targets).float().mean().item()
print(f"Manual accuracy: {manual_accuracy:.4f}")
