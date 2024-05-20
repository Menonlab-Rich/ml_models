from unet.model import UNet
import torch
import torch.nn as nn
import functools


class _Debug(nn.Module):
    def __init__(self, **kwargs):
        super(_Debug, self).__init__()
        self.kwargs = kwargs

    def forward(self, x):
        print(f'shape: {x.shape}')
        for k, v in self.kwargs.items():
            print(f'{k}: {v}')
        return x

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):        
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        in_channels = in_channels * 2 # Concatenating the input and target images
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
         
# re-export unet model
Generator = UNet

def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    gen = Generator(in_channels=3, out_channels=3)
    disc = Discriminator(in_channels=3, out_channels=1)
    assert gen(x).shape == (1, 3, 256, 256), "Generator test failed"
    assert disc(x, y).shape == (1, 1, 30, 30), "Discriminator test failed"
    print("All tests passed")

if __name__ == "__main__":
    test()