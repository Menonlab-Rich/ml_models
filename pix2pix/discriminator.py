import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    '''
    A CNN block consists of a convolutional layer, 
    a batch normalization layer, 
    and a leaky ReLU activation function.
    '''

    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            # The paper uses a stride of 2 to downsample,
            # a kernel size of 4, and a padding of 1.
            # The paper also uses a padding mode of "reflect".
            nn.Conv2d(in_channels, out_channels, 4, stride=stride,
                      padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    '''
    The discriminator is a convolutional neural network.
    The network is composed of 4 CNN blocks and 1 output layer.
    The discriminator takes in two images, the input image and the target image.
    The input image is concatenated with the target image along the channel dimension.
    The concatenated images are passed through the discriminator to produce a prediction map.
    The prediction map is a 1x30x30 tensor of values between 0 and 1. Where 0 represents a fake image and 1 represents a real image.
    The discriminator attempts to classify the input image as real or fake.
    
    See https://arxiv.org/pdf/1611.07004.pdf for more details.
    '''

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]) -> None:
        super().__init__()
        # the initial layer of the discriminator
        # does not have batch normalization.
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, features[0],
                kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),)

        layers = []
        # in_channels is the current number of features
        in_channels = features[0]
        stride = 2
        # skip the first element because it is used as the initial layer
        for feature in features[1:]:
            if feature == features[-1]:
                # the last layer of the discriminator
                # has a stride of 1 instead of 2.
                stride = 1
            layers.append(
                CNNBlock(
                    in_channels, feature, stride=stride
                )
            )
            in_channels = feature
            
        # the last layer of the discriminator produces a prediction map
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )

        # the model is a sequential model of the layers
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # concatenate the input images
        x = torch.cat([x, y], dim=1)
        # pass the concatenated images to the initial layer
        x = self.initial(x)
        # pass the output of the initial layer to the rest of the layers of the unet
        return self.model(x)


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)
    
if __name__ == "__main__":
    test()