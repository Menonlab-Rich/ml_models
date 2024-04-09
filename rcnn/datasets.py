import torch
import cv2
import numpy as np
import os
import glob as glob
from xml.etree import ElementTree as et
from config import (
    CLASSES, RESIZE_TO, TRAIN_IMG_DIR, TRAIN_ANNO_DIR, VALID_IMG_DIR, VALID_ANNO_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform
import logging
# the dataset class
class CustomDataset(Dataset):
    def __init__(self, img_path, ann_path, resize, classes, transforms=None):
        self.transforms = transforms
        self.img_path = img_path
        self.ann_path = ann_path
        self.resize = resize
        self.classes = classes
        
        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.img_path}/*.jpg")
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths if self._validate_xml(image_path.split(os.path.sep)[-1])]
        self.all_images = sorted(self.all_images)
        
    def _validate_xml(self, image_name):
        try:
            xml_filename = image_name[:-4] + '.xml'
            xml_file_path = os.path.join(self.ann_path, xml_filename)
            tree = et.parse(xml_file_path)
            root = tree.getroot()
            if len(root.findall('object')) == 0:
                return False
            for member in root.findall('object'):
                if member.find('name').text not in self.classes:
                    return False
            return True
        except Exception as e:
            logging.warning(f"Error validating XML file for {image_name}: {str(e)}")
            return False
    
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.img_path, image_name)
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = self._resize_image(image)
        image_resized /= 255.0
        
        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.ann_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(member.find('name').text))
            
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            yamx_final = (ymax/image_height)*self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # check if boxes is empty
        if len(boxes) == 0:
            # Set boxes to a tensor with shape (0, 4)
            logging.warning(f"No bounding boxes found for {image_name}")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            # bounding box to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # area of the bounding boxes
        if boxes.nelement() == 0:
            # If boxes is empty, set area to a tensor with shape (0,)
            logging.warning(f"No bounding boxes found for {image_name}")
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # no crowd instances
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target
    def __len__(self):
        return len(self.all_images)
    
    def _resize_image(self, image):
        # If self.resize is a tuple of 2 integers, it should be (height, width).
        # If self.resize is an int, the longer side of the image will be resized to this value, preserving the aspect ratio.
        # If self.resize is a float between 0 and 1, the image will be resized by this factor.
        # If self.resize is None, the image will not be resized.
        if isinstance(self.resize, int):
            longest_side = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
            scale = self.resize / longest_side
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
        elif isinstance(self.resize, float):
            width = int(image.shape[1] * self.resize)
            height = int(image.shape[0] * self.resize)
        elif isinstance(self.resize, tuple):
            width = self.resize[1]
            height = self.resize[0]
        else:
            self.width = image.shape[1]
            self.height = image.shape[0]
            return image
        
        self.width = width
        self.height = height
        return cv2.resize(image, (width, height))
# prepare the final datasets and data loaders
def create_train_dataset():
    train_dataset = CustomDataset(TRAIN_IMG_DIR, TRAIN_ANNO_DIR, RESIZE_TO, CLASSES, get_train_transform())
    return train_dataset
def create_valid_dataset():
    valid_dataset = CustomDataset(VALID_IMG_DIR, VALID_ANNO_DIR, RESIZE_TO, CLASSES, get_valid_transform())
    return valid_dataset
def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader
# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_IMG_DIR, TRAIN_ANNO_DIR, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        from config import OUT_DIR
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        try:
            cv2.imshow('Image', image)
            cv2.waitKey(0)
        except Exception as e:
            print("Saving the image as 'sample.jpg' due to error in visualization.")
            cv2.imwrite(os.path.join(OUT_DIR, 'sample.jpg'), image)
            input("Press Enter to continue...")
        
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)