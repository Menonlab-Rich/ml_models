import numpy as np
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomCocoDetection(Dataset):
    def __init__(self, image_dir, annotations_file, transforms=None, **kwargs):
        self._dataset = CocoDetection(root=image_dir, annFile=annotations_file, **kwargs)
        self._transforms = transforms

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        image, target = self._dataset[idx]
        image_np = np.array(image)

        # Check if there are annotations
        if target:
            bboxes = [box['bbox'] for box in target]
            labels = [box['category_id'] for box in target]
        else:
            # Default values for images without annotations
            bboxes = []
            labels = []

        if self._transforms:
            # Apply transformations both to images with and without annotations
            transformed = self._transforms(image=image_np, bboxes=bboxes, labels=labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_labels = transformed['labels']
            
            # Convert the transformed image and annotations back to CoCo format
            transformed_targets = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(transformed_bboxes, transformed_labels)]
        else:
            transformed_image = ToTensorV2()(image=image_np)['image']  # Convert to tensor
            transformed_targets = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(bboxes, labels)]

        return transformed_image, transformed_targets
