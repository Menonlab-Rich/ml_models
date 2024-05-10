import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F


class JML1(nn.Module):
    '''
    A soft Jaccard loss that uses the L1 norm to calculate the error
    Based on the paper Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels
    https://arxiv.org/pdf/2302.05666
    '''
    def __init__(self, num_classes=2, weights=None, smoothing=0.1, k=3):
        super(JML1, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        if weights is not None:
            # assume is weights is not none then it is a tensor
            self.register_buffer('weights', weights)
        self.k = k

    def forward(self, y_pred, y_true):
        # For boundary detection
        maxpool = torch.nn.MaxPool2d(
            kernel_size=self.k, stride=1, padding=self.k//2)
        # Ensure y_pred is in probability form
        y_pred = F.softmax(y_pred, dim=1)
        # One-hot encode y_true, expected y_true to be [batch_size, height, width] for image data
        y_true = F.one_hot(y_true, num_classes=self.num_classes).float()

        # OneHot moves the channel dimension to the last dimension
        # SoftMax does not change the order of the dimensions
        # So we change the one-hot encoded, y_true tensor to have the same order as the softmaxed y_pred
        y_true = y_true.permute(0, 3, 2, 1)
        y_true_pooled = maxpool(y_true)
        boundaries = (y_true_pooled > y_true).any(dim=1, keepdim=True)
        smoothed_boundaries = y_true * (1 - self.smoothing) + (
            self.smoothing / self.num_classes)
        labels = torch.where(boundaries, smoothed_boundaries, y_true)

        # Normalize over the spatial dimensions
        labels_norm = torch.norm(labels, p=1, dim=[2, 3])
        pred_norm = torch.norm(y_pred, p=1, dim=[2, 3])
        err_norm = torch.norm(labels - y_pred, p=1, dim=[2, 3])

        jaccard_index = (
            labels_norm + pred_norm - err_norm) \
            / (labels_norm + pred_norm + err_norm)

        if getattr(self, 'weights', None) is not None:
            jaccard_index = self.weights.unsqueeze(0) * jaccard_index
        # The loss is 1 - the average Jaccard index over the batch and classes
        # Averaging over both batch and classes
        return 1 - torch.mean(jaccard_index)


class WeightedComboLoss(nn.Module):
    def __init__(self, loss_a: nn.Module, loss_b: nn.Module, alpha=0.5,
                 beta=0.5) -> None:
        super(WeightedComboLoss, self).__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        return self.loss_a(
            y_pred, y_true) * self.alpha + self.loss_b(
            y_pred, y_true) * self.beta


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
        # Averaging over both batch and classes
        return 1 - torch.mean(jaccard_index)
