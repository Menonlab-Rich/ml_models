import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, w: int, h: int, nclasses: int) -> None:
        # initialize the parent class
        super(CNN, self).__init__() 
        if type(w) != int or type(h) != int or type(nclasses) != int:
            raise TypeError("w, h, and nclasses must be integers")
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        
        # 1x1 convolution to get the desired number of classes
        self.classifier = nn.Conv2d(3, nclasses, kernel_size=1) 

        
    def forward(self, x):
        x = self.features(x)
        x = self.deconv(x)
        x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    model = CNN(256, 256, 2)
    y = model(x)
    assert y.shape == (1, 2, 256, 256), f"Output shape is incorrect, expected {(1, 2, 256, 256)}, got {y.shape}"