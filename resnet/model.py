import torchvision.models as models
from torch import nn

def get_model(n_classes, freeze_layers=False, n_channels=3, multi_gpu=False):
    resnet = models.resnet50(pretrained=True)
    if freeze_layers:
        for param in resnet.parameters():
            param.requires_grad = False

    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, n_classes)
    
    if n_channels != 3:
        # Change the first layer to accept n_channels
        resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Use DataParallel to split the model across multiple GPUs
    if multi_gpu:
        resnet = nn.DataParallel(resnet)
    return resnet
