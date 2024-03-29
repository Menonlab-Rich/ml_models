import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import backbone_with_fpn
from torchvision.models import mobilenet_v3_large

def create_model(num_classes=3):  # Include background in the class count
    # Load a MobileNetV3-Large model pre-trained on ImageNet
    mobilenet_backbone = mobilenet_v3_large(pretrained=True)
    # Remove the classifier, as it's not used in Faster R-CNN
    backbone = mobilenet_backbone.features
    # Set the number of output channels in the backbone
    backbone.out_channels = 960
    
    # FPN requires to know the layers of the backbone which will be used
    return_layers = {'0': 0, '1': 1, '2': 2, '3': 3}  # This is an example, adjust if necessary

    # Create an FPN backbone
    backbone_fpn = backbone_with_fpn(backbone, 
                                     out_channels=256,
                                     returned_layers=[0, 1, 2, 3],
                                     extra_blocks=None)
    
    # Instantiate the Faster R-CNN model using the custom backbone
    model = FasterRCNN(backbone_fpn, 
                       num_classes=num_classes,
                       min_size=800, max_size=1333,
                       image_mean=[0.485, 0.456, 0.406],
                       image_std=[0.229, 0.224, 0.225])

    return model
