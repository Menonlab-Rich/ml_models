'''
An implementation of a simple autoencoder model using PyTorch.
Author: Rich Baird (2024)
'''

import torch
from torch import nn
from torch.nn import functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_channels, embedding_dim, rescale_factor=1.0, size=None):
        '''
        A simple autoencoder model
        Args:
            input_channels: The number of channels in the input image
            embedding_dim: The size of the embedding space
            rescale_factor: The factor by which to rescale the input image before encoding
                default: 1.0 (no rescaling)
        '''
        super(Autoencoder, self).__init__()
        if size is not None:
            self.register_buffer('size', torch.tensor(size, dtype=torch.long))
        self.register_buffer('rescale_factor', torch.tensor(rescale_factor, dtype=torch.float32))
        self._encoder_model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self._decoder_model = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        original_size = x.shape[-2:]
        if hasattr(self, 'size'):
            mode = 'bilinear' if self.size[-1] > x.shape[-1] else 'nearest'
            x = F.interpolate(x, size=list(self.size), mode=mode)
        else:
            mode = 'bilinear' if self.rescale_factor > 1.0 else 'nearest'
            x = F.interpolate(x, scale_factor=self.rescale_factor, mode=mode)
        embedding = self._encoder_model(x)
        reconstruction = self._decoder_model(embedding)
        if hasattr(self, 'size'):
            reconstruction = F.interpolate(reconstruction, size=original_size, mode=mode)
        return embedding, reconstruction
    
    @property
    def encoder(self):
        return self._encoder_model
    
    @property
    def decoder(self):
        return self._decoder_model
    
if __name__ == '__main__':
    model = Autoencoder(1, 16)
    print(model)
    x = torch.randn(1, 1, 128, 128)
    embedding, reconstruction = model(x)
    print(embedding.shape, reconstruction.shape)
    print('Success!')