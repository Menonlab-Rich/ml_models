import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV2AF(torch.nn.Module):
    '''
        A custom MobileNetV2 model for learning to autofocus.
        Based on the torchvision.models.mobilenet_v2 model.
        
        The modifications have been made according to the Learning to Autofocus paper.
        [https://doi.org/10.48550/arXiv.2004.12260](https://doi.org/10.48550/arXiv.2004.12260)
        '''
    def __init__(self, num_input_channels, num_classes=1000):
        super(MobileNetV2AF, self).__init__()
        width_mult = 4.0 # Width multiplier for the model
        original_model = models.mobilenet_v2(pretrained=False, width_mult=width_mult, num_classes=num_classes)
        self.features = original_model.features

        # Modify the first convolution layer
        # Original first layer: torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.features[0][0] = torch.nn.Conv2d(num_input_channels, 32 * width_mult, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Keeping features as is, except the modified first layer
        # Adjusting classifier for the number of classes
        self.classifier = torch.nn.Linear(original_model.last_channel, num_classes)

    def forward(self, x):
        '''
            Forward pass for the model.
            Identical to the forward pass of the original model.
        '''
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    # Test the model
    model = MobileNetV2AF(num_input_channels=1, num_classes=10)
    print(model)
    input = torch.randn(1, 1, 224, 224)
    output = model(input)
    print(output.size())