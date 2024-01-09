import torch
import torch.nn as nn
from enum import Enum


class Direction(Enum):
    down = 0
    up = 1


class block(nn.Module):
    def __init__(
            self, in_channels, out_channels, direction=Direction.down,
            activation="relu", dropout=False, norm=True):
        '''
        Creates a block of a convolutional neural network.
        The block consists of a convolutional layer,
        a batch normalization layer,
        and an activation function.
        The block can be used to downsample or upsample an image.
        The block can also be used to add dropout.
        
        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        direction : Direction
            The direction of the block.
            The direction can be either down or up.
            The default direction is down.
        activation : str
            The activation function.
            The activation function can be either "relu" or "leaky_relu".
            The default activation function is "relu".
        dropout : bool
            Whether to add dropout.
            The default is False.
        norm : bool
            Whether to add batch normalization.
        '''
        super().__init__()
        self.dropout = dropout
        layers = []
        if direction == Direction.down:
            layers.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=4,
                    stride=2, padding=1, padding_mode="reflect",
                    bias=False
                )
            )
        else:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=4,
                    stride=2, padding=1, bias=False
                )
            )
            
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == "relu":
            layers.append(nn.ReLU())
        else:
            layers.append(nn.LeakyReLU(0.2))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return nn.Dropout(0.5)(self.conv(x)) if self.dropout else self.conv(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64) -> None:
        super().__init__()
        # the initial layer does not have batch normalization.
        self.initial_down = block(in_channels, features, Direction.down, norm=False)
        
        # Downsample until we have just 2 features.
        for i in range(6):
            in_mult = min(2 ** i, 8)
            out_mult = min(2 ** (i + 1), 8)
            # Create a block and save it as an attribute of the model.
            b = block(features * in_mult, features * out_mult, Direction.down, activation="leaky")
            setattr(self, f"down_{i + 1}", b)        
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features * 8, features * 8, kernel_size=4,
                stride=2, padding=1, padding_mode="reflect"
            ),
            nn.ReLU(),
        )
        
        # Upsample until we get back to the original size.
        for i in range(7):
            in_mult = min(2 ** (7 - i), 8) * (2 if i > 0 else 1) # *2 because we concatenate the output of the previous layer. Except for the first layer.
            out_mult = min(2 ** (6 - i), 8)
            # Create a block and save it as an attribute of the model.
            b = block(features * in_mult, features * out_mult, Direction.up, activation="relu", dropout=True if i < 3 else False)
            setattr(self, f"up_{i + 1}", b)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, in_channels, kernel_size=4,
                stride=2, padding=1
            ),
            # The paper uses a tanh activation function for the last layer.
            # This means that the normalized values must be between -1 and 1.
            nn.Tanh(), 
            
        
        )
        
    def forward(self, x):
        '''
        Creates a forward pass through the generator.
        Uses a U-Net architecture.
        '''
        d1 = self.initial_down(x)
        d2 = self.down_1(d1)
        d3 = self.down_2(d2)
        d4 = self.down_3(d3)
        d5 = self.down_4(d4)
        d6 = self.down_5(d5)
        d7 = self.down_6(d6)
        bottleneck = self.bottleneck(d7)
        u1 = self.up_1(bottleneck)
        u2 = self.up_2(torch.cat([u1, d7], dim=1))
        u3 = self.up_3(torch.cat([u2, d6], dim=1))
        u4 = self.up_4(torch.cat([u3, d5], dim=1))
        u5 = self.up_5(torch.cat([u4, d4], dim=1))
        u6 = self.up_6(torch.cat([u5, d3], dim=1))
        u7 = self.up_7(torch.cat([u6, d2], dim=1))
        return self.final_up(torch.cat([u7, d1], dim=1))            
def test():
    x = torch.randn((1, 3, 256, 256))
    gen = Generator()
    print(gen(x).shape)
        
    
if __name__ == "__main__":
    test()