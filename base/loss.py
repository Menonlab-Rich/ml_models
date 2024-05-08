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
        self.weights = weights if weights is not None else torch.tensor(
            [1.0] * num_classes, dtype=torch.float32)
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


class CrossEntropyJaccardLoss(nn.Module):
    def __init__(self, jaccard: JaccardLoss, cross_entropy: nn.CrossEntropyLoss) -> None:
        super(CrossEntropyJaccardLoss, self).__init__()
        self.jaccard = jaccard
        self.cross_entropy = cross_entropy

    def forward(self, y_pred, y_true):
        return self.jaccard(y_pred, y_true) + self.cross_entropy(y_pred, y_true)


class PowerJaccardLoss(nn.Module):
    '''
    A power-weighted Jaccard loss, which is a generalization of the Jaccard loss
    Based on the paper: http://dx.doi.org/10.5220/0010304005610568
    '''
    def __init__(
            self, num_classes=2, weights=None, smoothing=1e-6, power=2,
            device='cuda'):
        super(PowerJaccardLoss, self).__init__()
        self.power = power
        self.smoothing = smoothing
        self.weights = weights if weights is not None else torch.tensor(
            [1.0] * num_classes, dtype=torch.float32).to(device)
        self.num_classes = num_classes
        self.register_buffer('weights', self.weights)

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = F.one_hot(
            y_pred, num_classes=self.num_classes).permute(
            0, 3, 1, 2)
        y_true = F.one_hot(
            y_true, num_classes=self.num_classes).permute(
            0, 3, 1, 2)

        intersection = torch.sum(y_true * y_pred, dim=[0, 2, 3])
        union = torch.sum(
            torch.pow(y_true, self.power) + torch.pow(y_pred, self.power),
            dim=[0, 2, 3]) - intersection

        jaccard = (intersection + self.smoothing) / (union + self.smoothing)

        # Apply weights
        weighted_jaccard = self.weights * jaccard
        # Return the mean Jaccard loss, weighted by class
        return 1 - torch.mean(weighted_jaccard)
