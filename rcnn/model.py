import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FastRCNNPredictor

def create_model(num_classes=3):  # Adjusting for 2 target classes + background
    # Load a pre-trained Faster R-CNN model with a MobileNetV3-Large-FPN backbone
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one (adjust for your num_classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model