import torchvision.models as models
from torch import nn

class SqueezeLaer(nn.Module):
    '''
    A custom layer to remove the extra dimension
    In cases where the channel dimension is 1, this layer will remove it
    It will do nothing if the channel dimension is greater than 1
    '''
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return x.squeeze()

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
    
    # Squeeze the output to remove the extra dimension
    resnet = nn.Sequential(resnet, SqueezeLaer()) 

    # Use DataParallel to split the model across multiple GPUs
    if multi_gpu:
        resnet = nn.DataParallel(resnet)
    
    return resnet
