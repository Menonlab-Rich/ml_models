import torch
import torch.nn as nn
import torchvision.transforms as tf
import warnings
import torch.nn.functional as F


class DoubleConvolution(nn.Module):
    '''
    Performs two convolution operations on the input
    '''

    def __init__(self, in_channels, out_channels):
        '''
        Creates a DoubleConvolution object

        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        '''
        super().__init__()
        # nn.Sequential() is used to sequentially apply a list of operations
        self.conv = nn.Sequential(
            # stride = 1 means same convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.float() # convert to float if not already
        return self.conv(x)


class UNet(nn.Module):
    '''
    UNet architecture
    '''

    def __init__(self, in_channels=3, out_channels=3,
                 features=[64, 128, 256, 512]) -> None:
        super().__init__()
        # Downward path
        self.down = nn.ModuleList()
        # Upward path
        self.up = nn.ModuleList()
        # Pooling down samples the input by a factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.down.append(DoubleConvolution(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.up.append(
                nn.ConvTranspose2d(
                    # feature*2 because of concatenation from skip connections
                    feature*2, feature, kernel_size=2, stride=2
                )
            )

            self.up.append(DoubleConvolution(feature*2, feature))

        # Create the bottleneck layer
        self.bottleneck = DoubleConvolution(features[-1], features[-1]*2)

        # Final convolution layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = x.float() # convert to float if not already
        if x.shape[2] > 256:
            warnings.warn(
                "x is larger than 256x256, this may cause memory issues")
        skip_connections = []
        for down in self.down:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # reverse the skip connections so that they are in the same order as the upward path
        skip_connections = skip_connections[::-1]

        # zip the skip connections and the transpose convolution layers of the upward path
        for i, (up_layer, skip_connection) in enumerate(
            zip(self.up[:: 2],
                skip_connections)):
            x = up_layer(x)
            x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            # concatenate the skip connection and the output of the transpose convolution layer
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # apply the double convolution layer
            x = self.up[i * 2 + 1](concat_skip)

        # apply the final convolution layer
        return self.final_conv(x)


def test():
    x = torch.randn((3, 3, 256, 256))
    model = UNet()
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

    x = torch.randn((3, 3, 485, 650))
    preds = model(x)
    assert preds.shape[2] == preds.shape[3] == 1024


if __name__ == '__main__':
    test()
