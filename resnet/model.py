import torchvision.models as models
from torch import nn
from config import MULTI_GPU

def get_model(num_classes, freeze_layers=False):
    resnet = models.resnet50(pretrained=True)
    if freeze_layers:
        for param in resnet.parameters():
            param.requires_grad = False

    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)
    if num_classes == 1:
        resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    # Use DataParallel to split the model across multiple GPUs
    if MULTI_GPU:
        resnet = nn.DataParallel(resnet)
    return resnet
