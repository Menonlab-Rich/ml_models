"""
A pytorch-lightning module for the UNet model.
Training, validation, and testing steps are copied from the original UNet implementation.
See: https://github.com/milesial/Pytorch-UNet/blob/master/train.py
"""

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim, nn
from pytorch_model import UNet
from base.loss import dice_loss
from base.metrics import ClassSpecificAccuracy






class UNetLightning(pl.LightningModule):
    def __init__(self, n_channels, n_classes, bilinear=False, learning_rate=1e-5, weight_decay=1e-8, momentum=0.999, amp=False):
        super(UNetLightning, self).__init__()
        self.model = UNet(n_channels, n_classes, bilinear)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.amp = amp
        self.criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
        self.train_accuracy = ClassSpecificAccuracy(ignore_indices=[0])  # 0 is the background class
        self.val_accuracy = ClassSpecificAccuracy(ignore_indices=[0])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(),
                                  lr=self.learning_rate, 
                                  weight_decay=self.weight_decay, 
                                  momentum=self.momentum,
                                  foreach=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        return {'optimizer': optimizer, 'lr_scheduler':{ "scheduler": scheduler, 'monitor': 'val_dice'}}

    def training_step(self, batch, batch_idx):
        images, true_masks, _ = batch
        masks_pred = self(images)
        
        if self.model.n_classes == 1:
            loss = self.criterion(masks_pred.squeeze(1), true_masks.float())
            loss += dice_loss(true_masks.float(), masks_pred.squeeze(1))
        else:
            loss = self.criterion(masks_pred, true_masks)
            loss += dice_loss(
                F.one_hot(true_masks, self.model.n_classes).permute(0, 3, 1, 2).float(),
                F.softmax(masks_pred, dim=1).float(),
                multiclass=True
            )
        
        # Update and log the custom accuracy
        self.train_accuracy(masks_pred, true_masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, true_masks = batch
        masks_pred = self(images)

        if self.model.n_classes == 1:
            loss = self.criterion(masks_pred.squeeze(1), true_masks.float())
            loss += dice_loss(true_masks.float(), masks_pred.squeeze(1))
            val_dice = -loss  # since dice_loss is negated
        else:
            loss = self.criterion(masks_pred, true_masks)
            loss += dice_loss(
                F.one_hot(true_masks, self.model.n_classes).permute(0, 3, 1, 2).float(),
                F.softmax(masks_pred, dim=1).float(),
                multiclass=True
            )
            val_dice = -loss  # since dice_loss is negated
        
        # Update and log the custom accuracy
        self.val_accuracy(masks_pred, true_masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_dice', val_dice, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_dice': val_dice}

    def test_step(self, batch, batch_idx):
        images, true_masks = batch
        masks_pred = self(images)

        if self.model.n_classes == 1:
            loss = self.criterion(masks_pred.squeeze(1), true_masks.float())
            loss += dice_loss(true_masks.float(), masks_pred.squeeze(1))
        else:
            loss = self.criterion(masks_pred, true_masks)
            loss += dice_loss(
                F.one_hot(true_masks, self.model.n_classes).permute(0, 3, 1, 2).float(),
                F.softmax(masks_pred, dim=1).float(),
                multiclass=True
            )

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def use_checkpointing(self):
        self.model.use_checkpointing()