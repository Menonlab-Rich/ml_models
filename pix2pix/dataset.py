from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config

class LineDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.files = [os.path.join(root_dir, file) for file in self.files]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # adding basename ensures that the fn works for relative and absolute paths
        img_path = os.path.join(self.root_dir, os.path.basename(self.files[idx]))
        img = Image.open(img_path)
        img = np.array(img)
        input_img = img[:, :256, :]
        target_img = img[:, 256:, :]
        augmentations = config.both_transform(image=input_img, image0=target_img)
        input_img, target_img = augmentations["image"], augmentations["image0"]
        
        input_img = config.transform_only_input(image=input_img)["image"]        
        target_img = config.transform_only_target(image=target_img)["image"]
        
        return input_img, target_img