import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, identity_downsample=None,
                 stride=1) -> None:
        ''''
        Create a block for ResNet

        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        identity_downsample: None or nn.Module
            Identity downsample layer to match the input and output dimensions
        stride: int
            Stride for the first convolution layer
        '''
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1,
            stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()  # Activation function
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, image_channels, num_classes) -> None:
        """
        Create a ResNet model

        Parameters:
        ----------
        block: nn.Module
            Block for the ResNet
        layers: list
            List containing the number of blocks for each layer
        image_channels: int
            Number of input channels
        num_classes: int
            Number of output classes
        """
        super(
            ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._expansion = 4
        self._out_layers = [64, 128, 256, 512]
        
        # Create the layers for the ResNet
        self._layer1 = self._make_layer(block, layers[0], self._out_layers[0], 1)
        self._layer2 = self._make_layer(block, layers[1], self._out_layers[1], 2)
        self._layer3 = self._make_layer(block, layers[2], self._out_layers[2], 2)
        self._layer4 = self._make_layer(block, layers[3], self._out_layers[3], 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self._out_layers[3] * self._expansion, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Forward pass through the layers
        for i in range(self._expansion):
            x = getattr(self, f'_layer{i+1}')(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def _make_layer(
            self, block, num_residual_blocks, out_channels, stride) -> nn.Module:
        """
        Create a layer for the ResNet

        Parameters:
        ----------
        block: nn.Module
            Block for the ResNet
        num_residual_blocks: int
            Number of residual blocks in the layer
        out_channels: int
            Number of output channels
        stride: int
            Stride for the first convolution layer
        """
        identity_downsample = None
        layers = []

        # If the stride is not 1 or the input channels are not equal to the output channels, then we need to downsample the identity
        # out_channels is multiplied by the expansion factor to match the dimensions of the residual block
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * self._expansion,
                    kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self._expansion))

        # Append the first residual block so that we can downsample the identity
        layers.append(
            block(
                self.in_channels, out_channels, identity_downsample,
                stride))

        # Update the input channels for the next residual block
        self.in_channels = out_channels * self._expansion

        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    @staticmethod
    def ResNet50(img_channels=3, num_classes=1000):
        return ResNet(Block, [3, 4, 6, 3], img_channels, num_classes)
    
    @staticmethod
    def ResNet101(img_channels=3, num_classes=1000):
        return ResNet(Block, [3, 4, 23, 3], img_channels, num_classes)
    
    @staticmethod
    def ResNet152(img_channels=3, num_classes=1000):
        return ResNet(Block, [3, 8, 36, 3], img_channels, num_classes)
    
if __name__ == "__main__":
    model = ResNet.ResNet50()
    x = torch.randn(2, 3, 224, 224)
    y = model(x).to('cpu')
    print(y.shape)
