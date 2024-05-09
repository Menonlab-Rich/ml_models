import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F


class JaccardLoss(nn.Module):
    def __init__(self, num_classes=2, weights=None, smoothing=1e-6):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        if weights is not None:
            self.register_buffer('weights', weights) # assume is weights is not none then it is a tensor

    def forward(self, y_pred, y_true):
        # Ensure y_pred is in probability form
        y_pred = F.softmax(y_pred, dim=1)
        # One-hot encode y_true, expected y_true to be [batch_size, height, width] for image data
        y_true = F.one_hot(y_true, num_classes=self.num_classes).float()
        
        # Change shape to [batch, classes, other dimensions...]
        y_pred = y_pred.permute(0, 2, 3, 1)
        y_true = y_true.permute(0, 2, 3, 1)


        # Flatten the last dimensions to simplify the sum operations
        y_pred = y_pred.contiguous().view(y_pred.shape[0], y_pred.shape[1], -1)
        y_true = y_true.contiguous().view(y_true.shape[0], y_true.shape[1], -1)

        # Calculate dot product and L1 norm across the spatial dimensions for each example in the batch
        dot_product = torch.sum(y_true * y_pred, dim=[0, 2, 3])
        l1_norm = torch.sum(torch.abs(y_pred - y_true), dim=[0, 2, 3])

        # Calculate the modified Jaccard index per example in the batch
        jaccard_index = dot_product / (dot_product + l1_norm + self.smoothing)

        if getattr(self, 'weights', None) is not None:
            jaccard_index = self.weights * jaccard_index
        # The loss is 1 - the average Jaccard index over the batch and classes
        return 1 - torch.mean(jaccard_index)  # Averaging over both batch and classes

class WeightedComboLoss(nn.Module):
    def __init__(self, loss_a: nn.Module, loss_b: nn.Module, alpha=0.5) -> None:
        super(WeightedComboLoss, self).__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        return self.loss_a(y_pred, y_true) * self.alpha + self.loss_b(y_pred, y_true)


class PowerJaccardLoss(nn.Module):
    def __init__(self, num_classes=2, weights=None, smoothing=1e-6, power=1.0):
        super(PowerJaccardLoss, self).__init__()
        self.num_classes = num_classes
        # If weights are not provided, use equal weighting
        if weights is not None:
            self.register_buffer('weights', weights)
        self.smoothing = smoothing
        self.power = power

    def forward(self, y_pred, y_true):
        # Ensure y_pred is in probability form
        y_pred = F.softmax(y_pred, dim=1)
        # One-hot encode y_true, expected y_true to be [batch_size, height, width] for image data
        y_true = F.one_hot(y_true, num_classes=self.num_classes).float()
        
        # Change shape to [batch, classes, other dimensions...]
        y_pred = y_pred.permute(0, 2, 3, 1)
        y_true = y_true.permute(0, 2, 3, 1)


        # Flatten the last dimensions to simplify the sum operations
        y_pred = y_pred.contiguous().view(y_pred.shape[0], y_pred.shape[1], -1)
        y_true = y_true.contiguous().view(y_true.shape[0], y_true.shape[1], -1)

        # Calculate dot product and L1 norm across the spatial dimensions for each example in the batch
        dot_product = torch.sum(y_true * y_pred, dim=2)
        l1_norm = torch.sum(torch.abs(y_pred - y_true), dim=2)

        # Calculate the modified Jaccard index per example in the batch
        jaccard_index = dot_product / (dot_product + l1_norm + self.smoothing)

        if getattr(self, 'weights', None) is not None:
            jaccard_index = self.weights * jaccard_index
        # The loss is 1 - the average Jaccard index over the batch and classes
        return 1 - torch.mean(jaccard_index)  # Averaging over both batch and classes
