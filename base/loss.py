import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Type, Union, Callable
import pytorch_lightning as pl

"""Common image segmentation losses.
"""

import torch

from torch.nn import functional as F


def bce_loss(true, logits, pos_weight=None):
    """Computes the weighted binary cross-entropy loss.

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, 1, H, W]. Corresponds to
            the raw output or logits of the model.
        pos_weight: a scalar representing the weight attributed
            to the positive class. This is especially useful for
            an imbalanced dataset.

    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        true.float(),
        pos_weight=pos_weight,
    )
    return bce_loss


def ce_loss(true, logits, weights, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weight: a tensor of shape [C,]. The weights attributed
            to each class.
        ignore: the class index to ignore.

    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        logits.float(),
        true.long(),
        ignore_index=ignore,
        weight=weights,
    )
    return ce_loss


def dice_loss(true: Tensor, logits: Tensor, eps=1e-7, multiclass=False):
    """Computes the Sørensen–Dice loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    device = true.device # Ensure the device is the same
    num_classes = logits.shape[1]
    true = true.long() # Ensure the true values are integers
    if not multiclass:
        assert num_classes == 1, "To perform a multiclass dice loss, set multiclass=True."
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1).to(device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).to(device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.

    Returns:
        tversky_loss: the Tversky loss.

    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff

    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)


def ce_dice(true, pred, log=False, w1=1, w2=1):
    pass


def ce_jaccard(true, pred, log=False, w1=1, w2=1):
    pass


def focal_loss(true, pred):
    pass

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

class MultiClassDiceLoss(nn.Module):
    def __init__(self, weights=None, smoothing=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        # If weights are not provided, use equal weighting
        if weights is not None:
            self.register_buffer('weights', weights)
        self.smoothing = smoothing

    def forward(self, y_pred, y_true):
        loss = dice_loss(y_true, y_pred, eps=self.smoothing)
        if getattr(self, 'weights', None) is not None:
            loss = self.weights * loss
            
            
class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None, scale=1.0):
        '''
        Create a weighted MSE loss function
        
        Parameters:
        ----------
        weights: torch.Tensor
            The weights to use for the loss
        scale: float
            The scale factor to apply to the loss
            Default: 1.0 (no scaling)
            
        Throws:
        -------
        AssertionError: If the input and target are not on the same device
        AssertionError: If the weights are provided and not on the same device as the input
        '''
        super(WeightedMSELoss, self).__init__()
        self.register_buffer('weight', weights)
        self.register_buffer('scale', torch.tensor(scale)) # Register the scale factor

    def forward(self, input, target, classes=None):
        assert input.device == target.device, 'Input and target must be on the same device'
        if self.weight is not None:
            assert self.weight.device == input.device, 'Weights must be on the same device as input'
            # Normalize the weights to sum to 1
            # map the classes to the weights
            if classes is not None:
                device = self.weight.device
                weights = torch.tensor(
                    [self.weight[cls] for cls in classes],
                    dtype=torch.float).to(device)  # Get the weights for the classes

            normalized_weight = weights / weights.sum()
            # Expand the weights to the same shape as the input for broadcasting
            normalized_weight = normalized_weight.view(-1, 1, 1, 1).expand_as(input)
            # Compute the weighted MSE
            loss = (normalized_weight * (input - target) ** 2).mean()
        else:
            # Fallback to regular MSE if no weights are provided
            loss = ((input - target) ** 2).mean()
        return loss * self.scale # Scale the loss by the scale factor

def with_loss_fn(loss_fn: Union[str, Callable, nn.Module],
                 **kwargs) -> Callable[[Type[pl.LightningModule]],
                                       Type[pl.LightningModule]]:
    '''
    A decorator to add a loss function to a LightningModule
    '''
    def decorator(cls: Type[pl.LightningDataModule]) -> Type[pl.LightningDataModule]:
        if isinstance(loss_fn, str):
            loss_fn_instance = globals()[loss_fn](**kwargs)
        else:
            loss_fn_instance = loss_fn(
                **kwargs) if callable(loss_fn) else loss_fn

        class WrappedClass(cls, pl.LightningModule):
            def __init__(self, *args, **init_kwargs):
                super(WrappedClass, self).__init__(*args, **init_kwargs)

            def loss_fn(self, *args, **kwargs):
                return loss_fn_instance(*args, **kwargs)

        return WrappedClass

    return decorator