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
        # If weights are not provided, use equal weighting
        self.weights = torch.tensor(weights if weights is not None else [
                                    1.0]*num_classes, dtype=torch.float32)
        # Register weights as a parameter if they need to be learnable, otherwise as a buffer
        if weights is not None and all(
                isinstance(w, torch.Tensor) for w in weights):
            self.weights = nn.Parameter(self.weights)
        else:
            self.register_buffer('weights', self.weights)
        self.smoothing = smoothing

    def forward(self, y_pred, y_true):
        # Ensure y_pred is in probability form
        y_pred = F.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = F.one_hot(
            y_pred, num_classes=self.num_classes).permute(
            0, 3, 1, 2)
        y_true = F.one_hot(
            y_true, num_classes=self.num_classes).permute(
            0, 3, 1, 2)

        intersection = torch.sum(y_true * y_pred, dim=[0, 2, 3])
        union = torch.sum(y_true + y_pred, dim=[0, 2, 3]) - intersection

        jaccard = (intersection + self.smoothing) / (union + self.smoothing)

        # Apply weights
        weighted_jaccard = self.weights * jaccard
        # Return the mean Jaccard loss, weighted by class
        return 1 - torch.mean(weighted_jaccard)


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
        if weights is not None and all(
                isinstance(w, torch.Tensor) for w in weights):
           self.register_buffer('weights', weights) 
        else:
            self.weights = weights
        self.smoothing = smoothing

    def forward(self, y_pred, y_true):
        # Ensure y_pred is in probability form
        y_pred = F.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = F.one_hot(
            y_pred, num_classes=self.num_classes).permute(
            0, 3, 1, 2)
        y_true = F.one_hot(
            y_true, num_classes=self.num_classes).permute(
            0, 3, 1, 2)

        intersection = torch.sum(y_true * y_pred, dim=[0, 2, 3])
        union = torch.sum(y_true + y_pred, dim=[0, 2, 3]) - intersection

        jaccard = (intersection + self.smoothing) / (union + self.smoothing)

        # Apply weights
        if self.weights is not None:
            weighted_jaccard = self.weights * jaccard
        else:
            weighted_jaccard = jaccard # If no weights are provided, use equal weighting
            
        # Return the mean Jaccard loss, weighted by class
        return 1 - torch.mean(weighted_jaccard)

