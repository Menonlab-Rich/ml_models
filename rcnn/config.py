import torch
import os

BATCH_SIZE = 16 # increase / decrease according to GPU memeory
# resize the image for training and transforms
# If this is a tuple of 2 integers, it should be (height, width).
# If this is an int, the longer side of the image will be resized to this value, preserving the aspect ratio.
# If this is a float between 0 and 1, the image will be resized by this factor.
# If this is None, the image will not be resized.
RESIZE_TO = 416 
NUM_EPOCHS = 10 # number of epochs to train for
NUM_WORKERS = 8 # number of worker processes for background data loading
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_IMG_DIR = r'D:\CZI_scope\code\data\videos\training_data\dfa_cmos\imgs\train'
TRAIN_ANNO_DIR = r'D:\CZI_scope\code\data\videos\training_data\dfa_cmos\voc_annotations\train'
# validation images and XML files directory
VALID_IMG_DIR = r'D:\CZI_scope\code\data\videos\training_data\dfa_cmos\imgs\val'
VALID_ANNO_DIR = r'D:\CZI_scope\code\data\videos\training_data\dfa_cmos\voc_annotations\val'
# classes: 0 index is reserved for background
CLASSES = [
    '__background__', '-1', '1'
]
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# whether to save annotated images
SAVE_ANNOTATED_IMAGES = True
# location to save model and plots
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')