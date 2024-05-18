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


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(
            self, input_channels, num_filters=64, num_layers=3,
            norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_channels (int)  -- the number of channels in input images
            num_filters (int)     -- the number of filters in the first conv layer
            num_layers (int)      -- the number of conv layers in the discriminator
            norm_layer            -- normalization layer
        """
        super(Discriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding_size = 1

        # Initial convolutional layer
        sequence = [
            nn.Conv2d(
                input_channels, num_filters, kernel_size=kernel_size, stride=2,
                padding=padding_size),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # Gradually increase the number of filters
        filter_mult_prev = 1

        for n in range(1, num_layers):
            filter_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    num_filters * filter_mult_prev, num_filters *
                    filter_mult, kernel_size=kernel_size, stride=2,
                    padding=padding_size, bias=use_bias),
                norm_layer(num_filters * filter_mult),
                nn.LeakyReLU(0.2, inplace=True),]
            filter_mult_prev = filter_mult

        # Final layers with no downsampling
        filter_mult_prev = filter_mult
        filter_mult = min(2 ** num_layers, 8)
        sequence += [
            nn.Conv2d(
                num_filters * filter_mult_prev, num_filters * filter_mult,
                kernel_size=kernel_size, stride=1, padding=padding_size,
                bias=use_bias),
            norm_layer(num_filters * filter_mult),
            nn.LeakyReLU(0.2, inplace=True)]

        # Output layer with 1 channel prediction map
        sequence += [nn.Conv2d(num_filters * filter_mult, 1,
                               kernel_size=kernel_size, stride=1, padding=padding_size)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input_tensor):
        """Standard forward."""
        return self.model(input_tensor)


# re-export unet model
Generator = UNet
