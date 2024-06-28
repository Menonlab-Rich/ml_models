"""
A pytorch-lightning module for the UNet model.
Training, validation, and testing steps are copied from the original UNet implementation.
See: https://github.com/milesial/Pytorch-UNet/blob/master/train.py
"""

import pytorch_lightning as pl
import torch.nn.functional as F
from torch import optim, nn
from pytorch_model import UNet
from base.loss import DiceLoss
from base.metrics import GeneralizedDiceScore
from torchmetrics.functional.segmentation import GeneralizedDiceScore
from torchmetrics import MeanMetric
from neptune.types import File

class UNetLightning(pl.LightningModule):
    def __init__(self, n_channels, n_classes, bilinear=False, learning_rate=1e-5, weight_decay=1e-8, momentum=0.999, amp=False):
        super(UNetLightning, self).__init__()
        self.model = UNet(n_channels, n_classes, bilinear, use_checkpointing=True)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.amp = amp
        self.n_classes = n_classes
        self.criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(ignore_index=0)  # 0 is the background class
        self.val_accuracy = GeneralizedDiceScore(self.n_classes, include_background=False)
        self.train_loss_metric = MeanMetric()
        self.val_loss_metric = MeanMetric()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(),
                                  lr=self.learning_rate, 
                                  weight_decay=self.weight_decay, 
                                  momentum=self.momentum,
                                  foreach=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': { "scheduler": scheduler, 'monitor': 'val_dice' }}
    
    def calc_loss(self, masks_pred, masks_true):
        if self.n_classes == 1:
            masks_pred = masks_pred.squeeze(1)
        criterion_loss = self.criterion(masks_pred, masks_true)
        dloss = self.dice_loss(masks_pred, masks_true)
        return criterion_loss + dloss

    def training_step(self, batch, batch_idx):
        images, true_masks, _ = batch
        masks_pred = self(images)
        loss = self.calc_loss(masks_pred, true_masks)
        
        # Update and log the training loss
        self.train_loss_metric.update(loss)
        self.log('train_loss', self.train_loss_metric, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, true_masks, _ = batch
        masks_pred = self(images)
        loss = self.calc_loss(masks_pred, true_masks)
        
        # Update and log the custom accuracy
        self.val_accuracy.update(masks_pred, true_masks)
        self.val_loss_metric.update(loss)
        self.log('val_loss', self.val_loss_metric, on_epoch=True, prog_bar=True)
        self.log('val_dice', self.val_accuracy, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_dice': self.val_accuracy}

    def validation_epoch_end(self, outputs):
        from random import sample
        # Reset the metrics after each validation epoch
        self.val_accuracy.reset()
        self.val_loss_metric.reset()
        
        # Select 3 random images from the validation set
        indices = sample(range(len(self.val_dataset)), 3)
        for i in indices:
            image, mask = self.val_dataset[i]
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            pred = self(image)
            self.plot_segmentation_map(image, mask, pred) # Log the segmentation map

    def test_step(self, batch, batch_idx):
        images, true_masks, _ = batch
        masks_pred = self(images)
        loss = self.calc_loss(masks_pred, true_masks)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def use_checkpointing(self):
        self.model.use_checkpointing()
    
    def _mask_to_rgb(self, mask):
        import numpy as np
        mask = mask.squeeze().cpu().numpy()
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
        mask_rgb[mask == 1] = [1, 0, 0]
        mask_rgb[mask == 2] = [0, 1, 0]
        mask_rgb *= 255
        return mask_rgb.as_type(np.uint8)
    
    def plot_segmentation_map(self, image, mask, pred):
        import numpy as np
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
        ax[0].set_title('Image')
        ax[1].imshow(self._mask_to_rgb(mask))
        ax[1].set_title('Ground Truth')
        ax[2].imshow(self._mask_to_rgb(pred))
        ax[2].set_title('Prediction')
        
        # Hide axes
        for a in ax:
            a.axis('off')
        # Add a label to the figure
        fig.suptitle('Segmentation Map')
        
        # Add legend to the figure
        legend = [plt.Line2D([0], [0], color='red', lw=4, label='625'),
                  plt.Line2D([0], [0], color='green', lw=4, label='605')]
        
        plt.tight_layout()
        self.logger.experiment['Segmentation Map'].log(File.as_image(fig))
        
        